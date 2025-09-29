#include "svm_classifier/multiclass_strategy.hpp"
#include "svm.h"        // libsvm
#include "linear.h"     // liblinear
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <numeric>      // for std::accumulate

namespace svm_classifier {

    namespace {
        // Helper function to setup SVM parameters
        svm_parameter setup_svm_parameters(const KernelParameters& params, int n_features) {
            svm_parameter svm_params;
            svm_params.svm_type = C_SVC;

            switch (params.get_kernel_type()) {
                case KernelType::RBF:
                    svm_params.kernel_type = RBF;
                    break;
                case KernelType::POLYNOMIAL:
                    svm_params.kernel_type = POLY;
                    break;
                case KernelType::SIGMOID:
                    svm_params.kernel_type = SIGMOID;
                    break;
                default:
                    throw std::runtime_error("Invalid kernel type for libsvm");
            }

            svm_params.degree = params.get_degree();
            svm_params.gamma = (params.get_gamma() == -1.0) ? 1.0 / n_features : params.get_gamma();
            svm_params.coef0 = params.get_coef0();
            svm_params.cache_size = params.get_cache_size();
            svm_params.eps = params.get_tolerance();
            svm_params.C = params.get_C();
            svm_params.nr_weight = 0;
            svm_params.weight_label = nullptr;
            svm_params.weight = nullptr;
            svm_params.nu = 0.5;
            svm_params.p = 0.1;
            svm_params.shrinking = 1;
            svm_params.probability = params.get_probability() ? 1 : 0;

            return svm_params;
        }

        // Helper function to setup linear parameters
        parameter setup_linear_parameters(const KernelParameters& params) {
            parameter linear_params;

            // Use solver that supports probability estimates if probability is requested
            if (params.get_probability()) {
                linear_params.solver_type = L2R_LR; // Logistic regression - supports probability
            } else {
                linear_params.solver_type = L2R_L2LOSS_SVC_DUAL; // Default solver for C-SVC
            }

            linear_params.C = params.get_C();
            linear_params.eps = params.get_tolerance();
            linear_params.nr_weight = 0;
            linear_params.weight_label = nullptr;
            linear_params.weight = nullptr;
            linear_params.p = 0.1;
            linear_params.nu = 0.5;
            linear_params.init_sol = nullptr;
            linear_params.regularize_bias = 1;
            linear_params.w_recalc = false;

            return linear_params;
        }
    } // anonymous namespace

    // OneVsRestStrategy Implementation
    OneVsRestStrategy::OneVsRestStrategy()
        : library_type_(SVMLibrary::LIBLINEAR)
    {
    }

    OneVsRestStrategy::~OneVsRestStrategy()
    {
        cleanup_models();
    }

    TrainingMetrics OneVsRestStrategy::fit(const torch::Tensor& X,
        const torch::Tensor& y,
        const KernelParameters& params,
        DataConverter& converter)
    {
        cleanup_models();

        // Validate input tensors
        converter.validate_tensors(X, y);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Store parameters and determine library type
        params_ = params;
        library_type_ = ::svm_classifier::get_svm_library(params.get_kernel_type());

        // Extract unique classes
        auto y_cpu = y.to(torch::kCPU);
        auto unique_classes_tensor = std::get<0>(at::_unique(y_cpu));
        classes_.clear();

        for (int i = 0; i < unique_classes_tensor.size(0); ++i) {
            classes_.push_back(unique_classes_tensor[i].item<int>());
        }

        std::sort(classes_.begin(), classes_.end());

        // Handle binary classification case
        if (classes_.size() <= 2) {
            // For binary classification, train a single classifier
            classes_.resize(2); // Ensure we have exactly 2 classes

            // Resize model vectors for binary classification
            if (library_type_ == SVMLibrary::LIBSVM) {
                svm_models_.resize(1);
            } else {
                linear_models_.resize(1);
            }

            auto binary_y = y;
            if (classes_.size() == 1) {
                // Edge case: only one class, create dummy binary problem
                classes_.push_back(classes_[0] + 1);
                binary_y = torch::cat({ y, torch::full({1}, classes_[1], y.options()) });
                auto dummy_x = torch::zeros({ 1, X.size(1) }, X.options());
                auto extended_X = torch::cat({ X, dummy_x });

                train_binary_classifier(extended_X, binary_y, params, converter, 0);
            } else {
                train_binary_classifier(X, binary_y, params, converter, 0);
            }
        } else {
            // Multiclass case: train one classifier per class
            if (library_type_ == SVMLibrary::LIBSVM) {
                svm_models_.resize(classes_.size());
            } else {
                linear_models_.resize(classes_.size());
            }
            data_converters_.resize(classes_.size());

            for (size_t i = 0; i < classes_.size(); ++i) {
                auto binary_y = create_binary_labels(y, classes_[i]);
                data_converters_[i] = std::make_unique<DataConverter>();
                train_binary_classifier(X, binary_y, params, *data_converters_[i], i);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        is_trained_ = true;

        TrainingMetrics metrics;
        metrics.training_time = duration.count() / 1000.0;
        metrics.status = TrainingStatus::SUCCESS;

        return metrics;
    }

    std::vector<int> OneVsRestStrategy::predict(const torch::Tensor& X, DataConverter& converter)
    {
        if (!is_trained_) {
            throw std::runtime_error("Model is not trained");
        }

        auto decision_values = decision_function(X, converter);
        std::vector<int> predictions;
        predictions.reserve(X.size(0));

        for (const auto& decision_row : decision_values) {
            // Find the class with maximum decision value
            auto max_it = std::max_element(decision_row.begin(), decision_row.end());
            int predicted_class_idx = std::distance(decision_row.begin(), max_it);
            predictions.push_back(classes_[predicted_class_idx]);
        }

        return predictions;
    }

    std::vector<std::vector<double>> OneVsRestStrategy::predict_proba(const torch::Tensor& X,
        DataConverter& converter)
    {
        if (!supports_probability()) {
            throw std::runtime_error("Probability prediction not supported for current configuration");
        }

        if (!is_trained_) {
            throw std::runtime_error("Model is not trained");
        }

        std::vector<std::vector<double>> probabilities;
        probabilities.reserve(X.size(0));

        size_t num_models = library_type_ == SVMLibrary::LIBSVM ? svm_models_.size() : linear_models_.size();

        for (int i = 0; i < X.size(0); ++i) {
            auto sample = X[i];
            std::vector<double> sample_probs;
            sample_probs.reserve(classes_.size());

            for (size_t j = 0; j < num_models; ++j) {
                sample_probs.push_back(get_sample_probability(sample, j, converter));
            }

            // For binary classification with single model, add complement probability
            if (classes_.size() == 2 && num_models == 1 && !sample_probs.empty()) {
                sample_probs.push_back(1.0 - sample_probs[0]);
            }

            // Normalize probabilities
            double sum = std::accumulate(sample_probs.begin(), sample_probs.end(), 0.0);
            if (sum > 0.0) {
                for (auto& prob : sample_probs) {
                    prob /= sum;
                }
            } else {
                // Uniform distribution if all probabilities are zero
                std::fill(sample_probs.begin(), sample_probs.end(), 1.0 / classes_.size());
            }

            probabilities.push_back(sample_probs);
        }

        return probabilities;
    }

    std::vector<std::vector<double>> OneVsRestStrategy::decision_function(const torch::Tensor& X,
        DataConverter& converter)
    {
        if (!is_trained_) {
            throw std::runtime_error("Model is not trained");
        }

        std::vector<std::vector<double>> decision_values;
        decision_values.reserve(X.size(0));

        size_t num_models = library_type_ == SVMLibrary::LIBSVM ? svm_models_.size() : linear_models_.size();

        for (int i = 0; i < X.size(0); ++i) {
            auto sample = X[i];
            std::vector<double> sample_decisions;
            sample_decisions.reserve(classes_.size());

            for (size_t j = 0; j < num_models; ++j) {
                sample_decisions.push_back(get_sample_decision_value(sample, j, converter));
            }

            // For binary classification with single model, add negative decision value
            if (classes_.size() == 2 && num_models == 1 && !sample_decisions.empty()) {
                sample_decisions.push_back(-sample_decisions[0]);
            }

            decision_values.push_back(sample_decisions);
        }

        return decision_values;
    }

    bool OneVsRestStrategy::supports_probability() const
    {
        if (!is_trained_) {
            return params_.get_probability();
        }

        // Check if any model supports probability
        if (library_type_ == SVMLibrary::LIBSVM) {
            for (const auto& model : svm_models_) {
                if (model && svm_check_probability_model(model.get())) {
                    return true;
                }
            }
        } else {
            for (const auto& model : linear_models_) {
                if (model && check_probability_model(model.get())) {
                    return true;
                }
            }
        }

        return false;
    }

    torch::Tensor OneVsRestStrategy::create_binary_labels(const torch::Tensor& y, int positive_class)
    {
        auto binary_labels = torch::ones_like(y) * (-1); // Initialize with -1 (negative class)
        auto positive_mask = (y == positive_class);
        binary_labels.masked_fill_(positive_mask, 1); // Set positive class to +1

        return binary_labels;
    }

    double OneVsRestStrategy::train_binary_classifier(const torch::Tensor& X,
        const torch::Tensor& y_binary,
        const KernelParameters& params,
        DataConverter& converter,
        int class_idx)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (library_type_ == SVMLibrary::LIBSVM) {
            // Use libsvm
            auto problem = converter.to_svm_problem(X, y_binary);
            auto svm_params = setup_svm_parameters(params, X.size(1));

            // Check parameters
            const char* error_msg = svm_check_parameter(problem.get(), &svm_params);
            if (error_msg) {
                throw std::runtime_error("SVM parameter error: " + std::string(error_msg));
            }

            // Train model
            auto model = svm_train(problem.get(), &svm_params);
            if (!model) {
                throw std::runtime_error("Failed to train SVM model");
            }

            svm_models_[class_idx] = std::unique_ptr<svm_model>(model);

        } else {
            // Use liblinear
            auto problem = converter.to_linear_problem(X, y_binary);
            auto linear_params = setup_linear_parameters(params);

            // Check parameters
            const char* error_msg = check_parameter(problem.get(), &linear_params);
            if (error_msg) {
                throw std::runtime_error("Linear parameter error: " + std::string(error_msg));
            }

            // Train model
            auto model = train(problem.get(), &linear_params);
            if (!model) {
                throw std::runtime_error("Failed to train linear model");
            }

            linear_models_[class_idx] = std::unique_ptr<::model>(model);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        return duration.count() / 1000.0;
    }

    double OneVsRestStrategy::get_sample_decision_value(const torch::Tensor& sample,
        size_t model_idx, DataConverter& converter) const
    {
        if (library_type_ == SVMLibrary::LIBSVM && svm_models_[model_idx]) {
            auto sample_node_vec = data_converters_.empty() ?
                converter.to_svm_node(sample) : data_converters_[model_idx]->to_svm_node(sample);
            double decision_value;
            svm_predict_values(svm_models_[model_idx].get(), sample_node_vec.data(), &decision_value);
            return decision_value;
        } else if (library_type_ == SVMLibrary::LIBLINEAR && linear_models_[model_idx]) {
            auto sample_node_vec = data_converters_.empty() ?
                converter.to_feature_node(sample) : data_converters_[model_idx]->to_feature_node(sample);
            double decision_value;
            predict_values(linear_models_[model_idx].get(), sample_node_vec.data(), &decision_value);
            return decision_value;
        }
        return 0.0;
    }

    double OneVsRestStrategy::get_sample_probability(const torch::Tensor& sample,
        size_t model_idx, DataConverter& converter) const
    {
        if (library_type_ == SVMLibrary::LIBSVM && svm_models_[model_idx]) {
            auto sample_node_vec = data_converters_.empty() ?
                converter.to_svm_node(sample) : data_converters_[model_idx]->to_svm_node(sample);
            double prob_estimates[2];
            svm_predict_probability(svm_models_[model_idx].get(), sample_node_vec.data(), prob_estimates);
            return prob_estimates[0]; // Probability of positive class
        } else if (library_type_ == SVMLibrary::LIBLINEAR && linear_models_[model_idx]) {
            auto sample_node_vec = data_converters_.empty() ?
                converter.to_feature_node(sample) : data_converters_[model_idx]->to_feature_node(sample);
            double prob_estimates[2];
            predict_probability(linear_models_[model_idx].get(), sample_node_vec.data(), prob_estimates);
            return prob_estimates[0]; // Probability of positive class
        }
        return 0.0;
    }

    void OneVsRestStrategy::cleanup_models()
    {
        for (auto& model : svm_models_) {
            if (model) {
                auto raw_model = model.release();
                svm_free_and_destroy_model(&raw_model);
            }
        }
        svm_models_.clear();

        for (auto& model : linear_models_) {
            if (model) {
                auto raw_model = model.release();
                free_and_destroy_model(&raw_model);
            }
        }
        linear_models_.clear();
        data_converters_.clear();

        is_trained_ = false;
    }

    // OneVsOneStrategy Implementation
    OneVsOneStrategy::OneVsOneStrategy()
        : library_type_(SVMLibrary::LIBLINEAR)
    {
    }

    OneVsOneStrategy::~OneVsOneStrategy()
    {
        cleanup_models();
    }

    TrainingMetrics OneVsOneStrategy::fit(const torch::Tensor& X,
        const torch::Tensor& y,
        const KernelParameters& params,
        DataConverter& converter)
    {
        cleanup_models();

        // Validate input tensors
        converter.validate_tensors(X, y);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Store parameters and determine library type
        params_ = params;
        library_type_ = ::svm_classifier::get_svm_library(params.get_kernel_type());

        // Extract unique classes
        auto y_cpu = y.to(torch::kCPU);
        auto unique_classes_tensor = std::get<0>(at::_unique(y_cpu));
        classes_.clear();

        for (int i = 0; i < unique_classes_tensor.size(0); ++i) {
            classes_.push_back(unique_classes_tensor[i].item<int>());
        }

        std::sort(classes_.begin(), classes_.end());

        // Generate all class pairs
        class_pairs_.clear();
        for (size_t i = 0; i < classes_.size(); ++i) {
            for (size_t j = i + 1; j < classes_.size(); ++j) {
                class_pairs_.emplace_back(classes_[i], classes_[j]);
            }
        }

        // Initialize model storage and data converters
        if (library_type_ == SVMLibrary::LIBSVM) {
            svm_models_.resize(class_pairs_.size());
        } else {
            linear_models_.resize(class_pairs_.size());
        }
        data_converters_.resize(class_pairs_.size());

        // Train one classifier for each class pair
        for (size_t i = 0; i < class_pairs_.size(); ++i) {
            auto [class1, class2] = class_pairs_[i];
            data_converters_[i] = std::make_unique<DataConverter>();
            train_pairwise_classifier(X, y, class1, class2, params, *data_converters_[i], i);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        is_trained_ = true;

        TrainingMetrics metrics;
        metrics.training_time = duration.count() / 1000.0;
        metrics.status = TrainingStatus::SUCCESS;

        return metrics;
    }

    std::vector<int> OneVsOneStrategy::predict(const torch::Tensor& X, DataConverter& converter)
    {
        if (!is_trained_) {
            throw std::runtime_error("Model is not trained");
        }

        auto decision_values = decision_function(X, converter);
        return vote_predictions(decision_values);
    }

    std::vector<std::vector<double>> OneVsOneStrategy::predict_proba(const torch::Tensor& X,
        DataConverter& converter)
    {
        // OvO probability estimation is more complex and typically done via
        // pairwise coupling (Hastie & Tibshirani, 1998)
        // For simplicity, we'll use decision function values and normalize

        auto decision_values = decision_function(X, converter);
        std::vector<std::vector<double>> probabilities;
        probabilities.reserve(X.size(0));

        for (const auto& decision_row : decision_values) {
            std::vector<double> class_scores(classes_.size(), 0.0);

            // Aggregate decision values for each class
            for (size_t i = 0; i < class_pairs_.size(); ++i) {
                auto [class1, class2] = class_pairs_[i];
                double decision = decision_row[i];

                auto it1 = std::find(classes_.begin(), classes_.end(), class1);
                auto it2 = std::find(classes_.begin(), classes_.end(), class2);

                if (it1 != classes_.end() && it2 != classes_.end()) {
                    size_t idx1 = std::distance(classes_.begin(), it1);
                    size_t idx2 = std::distance(classes_.begin(), it2);

                    if (decision > 0) {
                        class_scores[idx1] += 1.0;
                    } else {
                        class_scores[idx2] += 1.0;
                    }
                }
            }

            // Convert scores to probabilities
            double sum = std::accumulate(class_scores.begin(), class_scores.end(), 0.0);
            if (sum > 0.0) {
                for (auto& score : class_scores) {
                    score /= sum;
                }
            } else {
                std::fill(class_scores.begin(), class_scores.end(), 1.0 / classes_.size());
            }

            probabilities.push_back(class_scores);
        }

        return probabilities;
    }

    std::vector<std::vector<double>> OneVsOneStrategy::decision_function(const torch::Tensor& X,
        DataConverter& converter)
    {
        if (!is_trained_) {
            throw std::runtime_error("Model is not trained");
        }

        std::vector<std::vector<double>> decision_values;
        decision_values.reserve(X.size(0));

        for (int i = 0; i < X.size(0); ++i) {
            auto sample = X[i];
            std::vector<double> sample_decisions;
            sample_decisions.reserve(class_pairs_.size());

            for (size_t j = 0; j < class_pairs_.size(); ++j) {
                sample_decisions.push_back(get_sample_decision_value(sample, j));
            }

            decision_values.push_back(sample_decisions);
        }

        return decision_values;
    }

    bool OneVsOneStrategy::supports_probability() const
    {
        return params_.get_probability();
    }

    std::pair<torch::Tensor, torch::Tensor> OneVsOneStrategy::extract_binary_data(const torch::Tensor& X,
        const torch::Tensor& y,
        int class1,
        int class2)
    {
        auto mask = (y == class1) | (y == class2);
        auto filtered_X = X.index_select(0, torch::nonzero(mask).squeeze());
        auto filtered_y = y.index_select(0, torch::nonzero(mask).squeeze());

        // Convert to binary labels: class1 -> +1, class2 -> -1
        auto binary_y = torch::where(filtered_y == class1, torch::ones_like(filtered_y), torch::full_like(filtered_y, -1));

        return std::make_pair(filtered_X, binary_y);
    }

    double OneVsOneStrategy::train_pairwise_classifier(const torch::Tensor& X,
        const torch::Tensor& y,
        int class1,
        int class2,
        const KernelParameters& params,
        DataConverter& converter,
        int model_idx)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [filtered_X, binary_y] = extract_binary_data(X, y, class1, class2);

        if (library_type_ == SVMLibrary::LIBSVM) {
            // Use libsvm
            auto problem = converter.to_svm_problem(filtered_X, binary_y);
            auto svm_params = setup_svm_parameters(params, filtered_X.size(1));

            // Check parameters
            const char* error_msg = svm_check_parameter(problem.get(), &svm_params);
            if (error_msg) {
                throw std::runtime_error("SVM parameter error: " + std::string(error_msg));
            }

            // Train model
            auto model = svm_train(problem.get(), &svm_params);
            if (!model) {
                throw std::runtime_error("Failed to train SVM model");
            }

            svm_models_[model_idx] = std::unique_ptr<svm_model>(model);
        } else {
            // Use liblinear
            auto problem = converter.to_linear_problem(filtered_X, binary_y);
            auto linear_params = setup_linear_parameters(params);

            // Check parameters
            const char* error_msg = check_parameter(problem.get(), &linear_params);
            if (error_msg) {
                throw std::runtime_error("Linear parameter error: " + std::string(error_msg));
            }

            // Train model
            auto model = train(problem.get(), &linear_params);
            if (!model) {
                throw std::runtime_error("Failed to train linear model");
            }

            linear_models_[model_idx] = std::unique_ptr<::model>(model);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        return duration.count() / 1000.0;
    }

    double OneVsOneStrategy::get_sample_decision_value(const torch::Tensor& sample, size_t model_idx) const
    {
        if (library_type_ == SVMLibrary::LIBSVM && svm_models_[model_idx]) {
            auto sample_node_vec = data_converters_[model_idx]->to_svm_node(sample);
            double decision_value;
            svm_predict_values(svm_models_[model_idx].get(), sample_node_vec.data(), &decision_value);
            return decision_value;
        } else if (library_type_ == SVMLibrary::LIBLINEAR && linear_models_[model_idx]) {
            auto sample_node_vec = data_converters_[model_idx]->to_feature_node(sample);
            double decision_value;
            predict_values(linear_models_[model_idx].get(), sample_node_vec.data(), &decision_value);
            return decision_value;
        }
        return 0.0;
    }

    std::vector<int> OneVsOneStrategy::vote_predictions(const std::vector<std::vector<double>>& decisions)
    {
        std::vector<int> predictions;
        predictions.reserve(decisions.size());

        for (const auto& decision_row : decisions) {
            std::vector<int> votes(classes_.size(), 0);

            // Count votes from pairwise decisions
            for (size_t i = 0; i < class_pairs_.size(); ++i) {
                auto [class1, class2] = class_pairs_[i];
                double decision = decision_row[i];

                auto it1 = std::find(classes_.begin(), classes_.end(), class1);
                auto it2 = std::find(classes_.begin(), classes_.end(), class2);

                if (it1 != classes_.end() && it2 != classes_.end()) {
                    size_t idx1 = std::distance(classes_.begin(), it1);
                    size_t idx2 = std::distance(classes_.begin(), it2);

                    if (decision > 0) {
                        votes[idx1]++;
                    } else {
                        votes[idx2]++;
                    }
                }
            }

            // Find class with most votes
            auto max_it = std::max_element(votes.begin(), votes.end());
            int predicted_class_idx = std::distance(votes.begin(), max_it);
            predictions.push_back(classes_[predicted_class_idx]);
        }

        return predictions;
    }

    void OneVsOneStrategy::cleanup_models()
    {
        for (auto& model : svm_models_) {
            if (model) {
                auto raw_model = model.release();
                svm_free_and_destroy_model(&raw_model);
            }
        }
        svm_models_.clear();

        for (auto& model : linear_models_) {
            if (model) {
                auto raw_model = model.release();
                free_and_destroy_model(&raw_model);
            }
        }
        linear_models_.clear();
        data_converters_.clear();

        is_trained_ = false;
    }

    // Factory function
    std::unique_ptr<MulticlassStrategyBase> create_multiclass_strategy(MulticlassStrategy strategy)
    {
        switch (strategy) {
            case MulticlassStrategy::ONE_VS_REST:
                return std::make_unique<OneVsRestStrategy>();
            case MulticlassStrategy::ONE_VS_ONE:
                return std::make_unique<OneVsOneStrategy>();
            default:
                throw std::invalid_argument("Unknown multiclass strategy");
        }
    }

} // namespace svm_classifier