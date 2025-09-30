#include "svm_classifier/svm_classifier.hpp"
#include "svm_classifier/config.h"

#include <cmath>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>

namespace svm_classifier {

namespace {
// Helper function to convert tensor to vector
std::vector<int> tensor_to_vector(const torch::Tensor& tensor) {
    auto tensor_cpu = tensor.to(torch::kCPU);
    std::vector<int> result;
    result.reserve(tensor_cpu.size(0));
    for (int i = 0; i < tensor_cpu.size(0); ++i) {
        result.push_back(tensor_cpu[i].item<int>());
    }
    return result;
}

// Helper function to calculate class metrics from confusion matrix for a single class
std::tuple<double, double, double> calculate_class_metrics(
    const std::vector<std::vector<int>>& confusion_matrix,
    int class_idx) {
    int n_classes = confusion_matrix.size();
    int tp = confusion_matrix[class_idx][class_idx];
    int fp = 0, fn = 0;

    for (int j = 0; j < n_classes; ++j) {
        if (class_idx != j) {
            fp += confusion_matrix[j][class_idx]; // False positives
            fn += confusion_matrix[class_idx][j]; // False negatives
        }
    }

    double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double f1 = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    return { precision, recall, f1 };
}

// Helper function to calculate fold boundaries for cross-validation
std::pair<int, int> calculate_fold_boundaries(int n_samples, int fold, int n_folds) {
    int fold_size = n_samples / n_folds;
    int remainder = n_samples % n_folds;

    int val_start = fold * fold_size + std::min(fold, remainder);
    int val_end = val_start + fold_size + (fold < remainder ? 1 : 0);

    return { val_start, val_end };
}
} // anonymous namespace

SVMClassifier::SVMClassifier() : is_fitted_(false), n_features_(0) {
    data_converter_ = std::make_unique<DataConverter>();
    initialize_multiclass_strategy();
}

SVMClassifier::SVMClassifier(const nlohmann::json& config) : SVMClassifier() {
    set_parameters(config);
    initialize_multiclass_strategy();
}

SVMClassifier::SVMClassifier(KernelType kernel, double C, MulticlassStrategy multiclass_strategy)
    : is_fitted_(false), n_features_(0) {
    params_.set_kernel_type(kernel);
    params_.set_C(C);
    params_.set_multiclass_strategy(multiclass_strategy);

    data_converter_ = std::make_unique<DataConverter>();
    initialize_multiclass_strategy();
}

SVMClassifier::~SVMClassifier() = default;

SVMClassifier::SVMClassifier(SVMClassifier&& other) noexcept
    : params_(std::move(other.params_)), multiclass_strategy_(std::move(other.multiclass_strategy_)),
      data_converter_(std::move(other.data_converter_)), is_fitted_(other.is_fitted_), n_features_(other.n_features_),
      training_metrics_(other.training_metrics_) {
    other.is_fitted_ = false;
    other.n_features_ = 0;
}

SVMClassifier& SVMClassifier::operator=(SVMClassifier&& other) noexcept {
    if (this != &other) {
        params_ = std::move(other.params_);
        multiclass_strategy_ = std::move(other.multiclass_strategy_);
        data_converter_ = std::move(other.data_converter_);
        is_fitted_ = other.is_fitted_;
        n_features_ = other.n_features_;
        training_metrics_ = other.training_metrics_;

        other.is_fitted_ = false;
        other.n_features_ = 0;
    }
    return *this;
}

TrainingMetrics SVMClassifier::fit(const torch::Tensor& X, const torch::Tensor& y) {
    validate_input(X, y, false);

    // Store number of features
    n_features_ = X.size(1);

    // Set gamma to auto if needed
    if (params_.get_gamma() == -1.0) {
        params_.set_gamma(1.0 / n_features_);
    }

    // Train the multiclass strategy
    training_metrics_ = multiclass_strategy_->fit(X, y, params_, *data_converter_);

    is_fitted_ = true;

    return training_metrics_;
}

torch::Tensor SVMClassifier::predict(const torch::Tensor& X) {
    validate_input(X, torch::Tensor(), true);

    auto predictions = multiclass_strategy_->predict(X, *data_converter_);
    return data_converter_->from_predictions(std::vector<double>(predictions.begin(), predictions.end()));
}

torch::Tensor SVMClassifier::predict_proba(const torch::Tensor& X) {
    if (!supports_probability()) {
        throw std::runtime_error("Probability prediction not supported. Set probability=true during training.");
    }

    validate_input(X, torch::Tensor(), true);

    auto probabilities = multiclass_strategy_->predict_proba(X, *data_converter_);
    return data_converter_->from_probabilities(probabilities);
}

torch::Tensor SVMClassifier::decision_function(const torch::Tensor& X) {
    validate_input(X, torch::Tensor(), true);

    auto decision_values = multiclass_strategy_->decision_function(X, *data_converter_);
    return data_converter_->from_decision_values(decision_values);
}

double SVMClassifier::score(const torch::Tensor& X, const torch::Tensor& y_true) {
    validate_input(X, y_true, true);

    auto predictions = predict(X);
    auto y_true_cpu = y_true.to(torch::kCPU);
    auto predictions_cpu = predictions.to(torch::kCPU);

    // Calculate accuracy
    auto correct = (predictions_cpu == y_true_cpu);
    return correct.to(torch::kFloat32).mean().item<double>();
}

EvaluationMetrics SVMClassifier::evaluate(const torch::Tensor& X, const torch::Tensor& y_true) {
    validate_input(X, y_true, true);

    auto predictions = predict(X);

    // Convert to std::vector for easier processing
    auto y_true_vec = tensor_to_vector(y_true);
    auto y_pred_vec = tensor_to_vector(predictions);

    EvaluationMetrics metrics;

    // Calculate accuracy
    metrics.accuracy = score(X, y_true);

    // Calculate confusion matrix
    metrics.confusion_matrix = calculate_confusion_matrix(y_true_vec, y_pred_vec);

    // Calculate precision, recall, and F1-score
    auto [precision, recall, f1] = calculate_metrics_from_confusion_matrix(metrics.confusion_matrix);
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.f1_score = f1;

    return metrics;
}

void SVMClassifier::set_parameters(const nlohmann::json& config) {
    params_.set_parameters(config);

    // Re-initialize multiclass strategy if strategy changed
    initialize_multiclass_strategy();

    // Reset fitted state if already fitted
    if (is_fitted_) {
        is_fitted_ = false;
        n_features_ = 0;
    }
}

nlohmann::json SVMClassifier::get_parameters() const {
    auto params = params_.get_parameters();

    // Add classifier-specific information
    params["is_fitted"] = is_fitted_;
    params["n_features"] = n_features_;
    params["n_classes"] = get_n_classes();
    params["svm_library"] = (get_svm_library() == SVMLibrary::LIBLINEAR) ? "liblinear" : "libsvm";

    return params;
}

int SVMClassifier::get_n_classes() const {
    if (!is_fitted_) {
        return 0;
    }
    return multiclass_strategy_->get_n_classes();
}

std::vector<int> SVMClassifier::get_classes() const {
    if (!is_fitted_) {
        return {};
    }
    return multiclass_strategy_->get_classes();
}

bool SVMClassifier::supports_probability() const {
    if (!is_fitted_) {
        return params_.get_probability();
    }
    return multiclass_strategy_->supports_probability();
}

void SVMClassifier::save_model(const std::string& filename) const {
    if (!is_fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    // For now, save parameters as JSON
    // Full model serialization would require more complex implementation
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    nlohmann::json model_data = { { "parameters", get_parameters() },
                                  { "training_metrics",
                                    { { "training_time", training_metrics_.training_time },
                                      { "support_vectors", training_metrics_.support_vectors },
                                      { "iterations", training_metrics_.iterations },
                                      { "objective_value", training_metrics_.objective_value } } },
                                  { "classes", get_classes() },
                                  { "version", "1.0" } };

    file << model_data.dump(2);
    file.close();
}

void SVMClassifier::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    nlohmann::json model_data;
    file >> model_data;
    file.close();

    // Load parameters
    if (model_data.contains("parameters")) {
        set_parameters(model_data["parameters"]);
    }

    // Load training metrics
    if (model_data.contains("training_metrics")) {
        auto tm = model_data["training_metrics"];
        training_metrics_.training_time = tm.value("training_time", 0.0);
        training_metrics_.support_vectors = tm.value("support_vectors", 0);
        training_metrics_.iterations = tm.value("iterations", 0);
        training_metrics_.objective_value = tm.value("objective_value", 0.0);
        training_metrics_.status = TrainingStatus::SUCCESS;
    }

    // Note: Full model loading would require serializing the actual SVM models
    // For now, this provides parameter persistence
    throw std::runtime_error("Full model loading not yet implemented. Only parameter loading is supported.");
}

std::vector<double> SVMClassifier::cross_validate(const torch::Tensor& X, const torch::Tensor& y, int cv) {
    validate_input(X, y, false);

    if (cv < 2) {
        throw std::invalid_argument("Number of folds must be >= 2");
    }

    std::vector<double> scores;
    scores.reserve(cv);

    // Store original fitted state
    bool was_fitted = is_fitted_;
    auto original_metrics = training_metrics_;

    for (int fold = 0; fold < cv; ++fold) {
        auto [X_train, y_train, X_val, y_val] = split_for_cv(X, y, fold, cv);

        // Create temporary classifier with same parameters
        SVMClassifier temp_clf(params_.get_parameters());

        // Train on training fold
        temp_clf.fit(X_train, y_train);

        // Evaluate on validation fold
        double fold_score = temp_clf.score(X_val, y_val);
        scores.push_back(fold_score);
    }

    // Restore original state
    is_fitted_ = was_fitted;
    training_metrics_ = original_metrics;

    return scores;
}

nlohmann::json
SVMClassifier::grid_search(const torch::Tensor& X, const torch::Tensor& y, const nlohmann::json& param_grid, int cv) {
    validate_input(X, y, false);

    auto param_combinations = generate_param_combinations(param_grid);

    double best_score = -1.0;
    nlohmann::json best_params;
    std::vector<double> all_scores;

    for (const auto& params : param_combinations) {
        SVMClassifier temp_clf(params);
        auto scores = temp_clf.cross_validate(X, y, cv);

        double mean_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        all_scores.push_back(mean_score);

        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = params;
        }
    }

    return { { "best_params", best_params }, { "best_score", best_score }, { "cv_results", all_scores } };
}

torch::Tensor SVMClassifier::get_feature_importance() const {
    if (!is_fitted_) {
        throw std::runtime_error("Model is not fitted");
    }

    if (params_.get_kernel_type() != KernelType::LINEAR) {
        throw std::runtime_error("Feature importance only available for linear kernels");
    }

    // This would require access to the linear model weights
    // Implementation depends on the multiclass strategy and would need
    // to extract weights from the underlying liblinear models
    throw std::runtime_error("Feature importance extraction not yet implemented");
}

void SVMClassifier::reset() {
    is_fitted_ = false;
    n_features_ = 0;
    training_metrics_ = TrainingMetrics();
    data_converter_->cleanup();
}

void SVMClassifier::validate_input(const torch::Tensor& X, const torch::Tensor& y, bool check_fitted) {
    if (check_fitted && !is_fitted_) {
        throw std::runtime_error(
            "This SVMClassifier instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator.");
    }

    data_converter_->validate_tensors(X, y);

    if (check_fitted && X.size(1) != n_features_) {
        throw std::invalid_argument(
            "Number of features in X (" + std::to_string(X.size(1)) +
            ") does not match number of features during training (" + std::to_string(n_features_) + ")");
    }
}

void SVMClassifier::initialize_multiclass_strategy() {
    multiclass_strategy_ = create_multiclass_strategy(params_.get_multiclass_strategy());
}

std::vector<std::vector<int>> SVMClassifier::calculate_confusion_matrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred) {
    // Get unique classes
    std::set<int> unique_classes;
    for (int label : y_true)
        unique_classes.insert(label);
    for (int label : y_pred)
        unique_classes.insert(label);

    std::vector<int> classes(unique_classes.begin(), unique_classes.end());
    std::sort(classes.begin(), classes.end());

    int n_classes = classes.size();
    std::vector<std::vector<int>> confusion_matrix(n_classes, std::vector<int>(n_classes, 0));

    // Create class to index mapping
    std::unordered_map<int, int> class_to_idx;
    for (size_t i = 0; i < classes.size(); ++i) {
        class_to_idx[classes[i]] = i;
    }

    // Fill confusion matrix
    for (size_t i = 0; i < y_true.size(); ++i) {
        int true_idx = class_to_idx[y_true[i]];
        int pred_idx = class_to_idx[y_pred[i]];
        confusion_matrix[true_idx][pred_idx]++;
    }

    return confusion_matrix;
}

std::tuple<double, double, double> SVMClassifier::calculate_metrics_from_confusion_matrix(
    const std::vector<std::vector<int>>& confusion_matrix) {
    int n_classes = confusion_matrix.size();
    if (n_classes == 0) {
        return { 0.0, 0.0, 0.0 };
    }

    std::vector<double> precision(n_classes), recall(n_classes), f1(n_classes);

    for (int i = 0; i < n_classes; ++i) {
        auto [p, r, f] = calculate_class_metrics(confusion_matrix, i);
        precision[i] = p;
        recall[i] = r;
        f1[i] = f;
    }

    // Calculate macro averages
    double macro_precision = std::accumulate(precision.begin(), precision.end(), 0.0) / n_classes;
    double macro_recall = std::accumulate(recall.begin(), recall.end(), 0.0) / n_classes;
    double macro_f1 = std::accumulate(f1.begin(), f1.end(), 0.0) / n_classes;

    return { macro_precision, macro_recall, macro_f1 };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SVMClassifier::split_for_cv(const torch::Tensor& X, const torch::Tensor& y, int fold, int n_folds) {
    int n_samples = X.size(0);
    auto [val_start, val_end] = calculate_fold_boundaries(n_samples, fold, n_folds);

    // Create indices
    auto all_indices = torch::arange(n_samples, torch::kLong);
    auto val_indices = all_indices.slice(0, val_start, val_end);

    // Training indices (everything except validation)
    auto train_indices = torch::cat({ all_indices.slice(0, 0, val_start), all_indices.slice(0, val_end, n_samples) });

    // Split data
    return { X.index_select(0, train_indices), y.index_select(0, train_indices), X.index_select(0, val_indices),
             y.index_select(0, val_indices) };
}

std::vector<nlohmann::json> SVMClassifier::generate_param_combinations(const nlohmann::json& param_grid) {
    std::vector<nlohmann::json> combinations;

    // Extract parameter names and values
    std::vector<std::string> param_names;
    std::vector<std::vector<nlohmann::json>> param_values;

    for (auto& [key, value] : param_grid.items()) {
        param_names.push_back(key);
        if (value.is_array()) {
            param_values.push_back(value);
        } else {
            param_values.push_back({ value });
        }
    }

    // Generate all combinations using recursive approach
    std::function<void(int, nlohmann::json&)> generate_combinations = [&](int param_idx,
                                                                          nlohmann::json& current_params) {
        if (static_cast<size_t>(param_idx) == param_names.size()) {
            combinations.push_back(current_params);
            return;
        }

        for (const auto& value : param_values[param_idx]) {
            current_params[param_names[param_idx]] = value;
            generate_combinations(param_idx + 1, current_params);
        }
    };

    nlohmann::json current_params;
    generate_combinations(0, current_params);

    return combinations;
}

std::string SVMClassifier::version() {
    return SVMCLASSIFIER_VERSION;
}

} // namespace svm_classifier