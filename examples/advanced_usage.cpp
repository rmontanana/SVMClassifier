#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace svm_classifier;
using json = nlohmann::json;

/**
 * @brief Generate a more realistic multi-class dataset with noise
 */
std::pair<torch::Tensor, torch::Tensor> generate_realistic_dataset(int n_samples,
    int n_features,
    int n_classes,
    double noise_factor = 0.1)
{
    torch::manual_seed(42);

    // Create class centers
    auto centers = torch::randn({ n_classes, n_features }) * 3.0;

    std::vector<torch::Tensor> class_data;
    std::vector<torch::Tensor> class_labels;

    int samples_per_class = n_samples / n_classes;

    for (int c = 0; c < n_classes; ++c) {
        // Generate samples around each class center
        auto class_samples = torch::randn({ samples_per_class, n_features }) * noise_factor;
        class_samples += centers[c].unsqueeze(0).expand({ samples_per_class, n_features });

        auto labels = torch::full({ samples_per_class }, c, torch::kInt32);

        class_data.push_back(class_samples);
        class_labels.push_back(labels);
    }

    // Concatenate all classes
    auto X = torch::cat(class_data, 0);
    auto y = torch::cat(class_labels, 0);

    // Shuffle the data
    auto indices = torch::randperm(X.size(0));
    X = X.index_select(0, indices);
    y = y.index_select(0, indices);

    return { X, y };
}

/**
 * @brief Normalize features to [0, 1] range
 */
torch::Tensor normalize_features(const torch::Tensor& X)
{
    auto min_vals = std::get<0>(torch::min(X, 0));
    auto max_vals = std::get<0>(torch::max(X, 0));
    auto range = max_vals - min_vals;

    // Avoid division by zero
    range = torch::where(range == 0.0, torch::ones_like(range), range);

    return (X - min_vals) / range;
}

/**
 * @brief Standardize features (zero mean, unit variance)
 */
torch::Tensor standardize_features(const torch::Tensor& X)
{
    auto mean = X.mean(0);
    auto std = X.std(0);

    // Avoid division by zero
    std = torch::where(std == 0.0, torch::ones_like(std), std);

    return (X - mean) / std;
}

/**
 * @brief Print detailed evaluation metrics
 */
void print_evaluation_metrics(const EvaluationMetrics& metrics, const std::string& title)
{
    std::cout << "\n=== " << title << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Accuracy:  " << metrics.accuracy * 100 << "%" << std::endl;
    std::cout << "Precision: " << metrics.precision * 100 << "%" << std::endl;
    std::cout << "Recall:    " << metrics.recall * 100 << "%" << std::endl;
    std::cout << "F1-Score:  " << metrics.f1_score * 100 << "%" << std::endl;

    std::cout << "\nConfusion Matrix:" << std::endl;
    for (size_t i = 0; i < metrics.confusion_matrix.size(); ++i) {
        for (size_t j = 0; j < metrics.confusion_matrix[i].size(); ++j) {
            std::cout << std::setw(6) << metrics.confusion_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Demonstrate comprehensive hyperparameter tuning
 */
void demonstrate_hyperparameter_tuning()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPREHENSIVE HYPERPARAMETER TUNING EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Generate dataset
    auto [X, y] = generate_realistic_dataset(1000, 20, 4, 0.3);

    // Standardize features
    X = standardize_features(X);

    std::cout << "Dataset: " << X.size(0) << " samples, " << X.size(1)
        << " features, " << torch::unique(y).size(0) << " classes" << std::endl;

    // Define comprehensive parameter grid
    json param_grids = {
        {
            "name", "Linear SVM Grid"
        },
        {
            "parameters", {
                {"kernel", {"linear"}},
                {"C", {0.01, 0.1, 1.0, 10.0, 100.0}},
                {"multiclass_strategy", {"ovr", "ovo"}}
            }
        }
    };

    SVMClassifier svm;

    std::cout << "\n--- Linear SVM Hyperparameter Search ---" << std::endl;
    auto linear_grid = param_grids["parameters"];
    auto linear_results = svm.grid_search(X, y, linear_grid, 5);

    std::cout << "Best Linear SVM parameters:" << std::endl;
    std::cout << linear_results["best_params"].dump(2) << std::endl;
    std::cout << "Best CV score: " << std::fixed << std::setprecision(4)
        << linear_results["best_score"].get<double>() * 100 << "%" << std::endl;

    // RBF parameter grid
    json rbf_grid = {
        {"kernel", {"rbf"}},
        {"C", {0.1, 1.0, 10.0}},
        {"gamma", {0.01, 0.1, 1.0, "auto"}},
        {"multiclass_strategy", {"ovr", "ovo"}}
    };

    std::cout << "\n--- RBF SVM Hyperparameter Search ---" << std::endl;
    auto rbf_results = svm.grid_search(X, y, rbf_grid, 3);

    std::cout << "Best RBF SVM parameters:" << std::endl;
    std::cout << rbf_results["best_params"].dump(2) << std::endl;
    std::cout << "Best CV score: " << std::fixed << std::setprecision(4)
        << rbf_results["best_score"].get<double>() * 100 << "%" << std::endl;

    // Polynomial parameter grid
    json poly_grid = {
        {"kernel", {"polynomial"}},
        {"C", {0.1, 1.0, 10.0}},
        {"degree", {2, 3, 4}},
        {"gamma", {0.01, 0.1, "auto"}},
        {"coef0", {0.0, 1.0}}
    };

    std::cout << "\n--- Polynomial SVM Hyperparameter Search ---" << std::endl;
    auto poly_results = svm.grid_search(X, y, poly_grid, 3);

    std::cout << "Best Polynomial SVM parameters:" << std::endl;
    std::cout << poly_results["best_params"].dump(2) << std::endl;
    std::cout << "Best CV score: " << std::fixed << std::setprecision(4)
        << poly_results["best_score"].get<double>() * 100 << "%" << std::endl;

    // Compare all models
    std::cout << "\n--- Model Comparison Summary ---" << std::endl;
    std::cout << std::setw(15) << "Model" << std::setw(12) << "CV Score" << std::setw(30) << "Best Parameters" << std::endl;
    std::cout << std::string(57, '-') << std::endl;

    std::cout << std::setw(15) << "Linear"
        << std::setw(12) << std::fixed << std::setprecision(4)
        << linear_results["best_score"].get<double>() * 100 << "%"
        << std::setw(30) << "C=" + std::to_string(linear_results["best_params"]["C"].get<double>()) << std::endl;

    std::cout << std::setw(15) << "RBF"
        << std::setw(12) << std::fixed << std::setprecision(4)
        << rbf_results["best_score"].get<double>() * 100 << "%"
        << std::setw(30) << "C=" + std::to_string(rbf_results["best_params"]["C"].get<double>()) << std::endl;

    std::cout << std::setw(15) << "Polynomial"
        << std::setw(12) << std::fixed << std::setprecision(4)
        << poly_results["best_score"].get<double>() * 100 << "%"
        << std::setw(30) << "deg=" + std::to_string(rbf_results["best_params"]["degree"].get<int>()) << std::endl;
}

/**
 * @brief Demonstrate model evaluation and validation
 */
void demonstrate_model_evaluation()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "MODEL EVALUATION AND VALIDATION EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Generate larger dataset for proper train/test split
    auto [X_full, y_full] = generate_realistic_dataset(2000, 15, 5, 0.2);

    // Normalize features
    X_full = normalize_features(X_full);

    // Train/test split (80/20)
    int n_train = static_cast<int>(X_full.size(0) * 0.8);
    auto X_train = X_full.slice(0, 0, n_train);
    auto y_train = y_full.slice(0, 0, n_train);
    auto X_test = X_full.slice(0, n_train);
    auto y_test = y_full.slice(0, n_train);

    std::cout << "Dataset split:" << std::endl;
    std::cout << "  Training: " << X_train.size(0) << " samples" << std::endl;
    std::cout << "  Testing:  " << X_test.size(0) << " samples" << std::endl;
    std::cout << "  Features: " << X_train.size(1) << std::endl;
    std::cout << "  Classes:  " << torch::unique(y_train).size(0) << std::endl;

    // Configure different models for comparison
    std::vector<json> model_configs = {
        {{"kernel", "linear"}, {"C", 1.0}, {"multiclass_strategy", "ovr"}},
        {{"kernel", "linear"}, {"C", 1.0}, {"multiclass_strategy", "ovo"}},
        {{"kernel", "rbf"}, {"C", 10.0}, {"gamma", 0.1}, {"multiclass_strategy", "ovr"}},
        {{"kernel", "rbf"}, {"C", 10.0}, {"gamma", 0.1}, {"multiclass_strategy", "ovo"}},
        {{"kernel", "polynomial"}, {"degree", 3}, {"C", 1.0}}
    };

    std::vector<std::string> model_names = {
        "Linear (OvR)",
        "Linear (OvO)",
        "RBF (OvR)",
        "RBF (OvO)",
        "Polynomial"
    };

    std::cout << "\n--- Training and Evaluating Models ---" << std::endl;

    for (size_t i = 0; i < model_configs.size(); ++i) {
        std::cout << "\n" << std::string(40, '-') << std::endl;
        std::cout << "Model: " << model_names[i] << std::endl;
        std::cout << "Config: " << model_configs[i].dump() << std::endl;

        SVMClassifier svm(model_configs[i]);

        // Train the model
        auto start_time = std::chrono::high_resolution_clock::now();
        auto training_metrics = svm.fit(X_train, y_train);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Training time: " << training_duration.count() << " ms" << std::endl;

        // Evaluate on training set
        auto train_metrics = svm.evaluate(X_train, y_train);
        print_evaluation_metrics(train_metrics, "Training Set Performance");

        // Evaluate on test set
        auto test_metrics = svm.evaluate(X_test, y_test);
        print_evaluation_metrics(test_metrics, "Test Set Performance");

        // Cross-validation
        std::cout << "\n--- Cross-Validation Results ---" << std::endl;
        auto cv_scores = svm.cross_validate(X_train, y_train, 5);

        double mean_cv = 0.0;
        for (double score : cv_scores) {
            mean_cv += score;
        }
        mean_cv /= cv_scores.size();

        double std_cv = 0.0;
        for (double score : cv_scores) {
            std_cv += (score - mean_cv) * (score - mean_cv);
        }
        std_cv = std::sqrt(std_cv / cv_scores.size());

        std::cout << "CV Scores: ";
        for (size_t j = 0; j < cv_scores.size(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << cv_scores[j];
            if (j < cv_scores.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "Mean CV: " << std::fixed << std::setprecision(4) << mean_cv * 100 << "% ± " << std_cv * 100 << "%" << std::endl;

        // Prediction analysis
        auto predictions = svm.predict(X_test);
        std::cout << "\n--- Prediction Analysis ---" << std::endl;

        // Count predictions per class
        auto unique_preds = torch::unique(predictions);
        std::cout << "Predicted classes: ";
        for (int j = 0; j < unique_preds.size(0); ++j) {
            std::cout << unique_preds[j].item<int>();
            if (j < unique_preds.size(0) - 1) std::cout << ", ";
        }
        std::cout << std::endl;

        // Test probability prediction if supported
        if (svm.supports_probability()) {
            std::cout << "Probability prediction: Supported" << std::endl;
            auto probabilities = svm.predict_proba(X_test.slice(0, 0, 5));  // First 5 samples
            std::cout << "Sample probabilities (first 5 samples):" << std::endl;
            for (int j = 0; j < 5; ++j) {
                std::cout << "  Sample " << j << ": ";
                for (int k = 0; k < probabilities.size(1); ++k) {
                    std::cout << std::fixed << std::setprecision(3)
                        << probabilities[j][k].item<double>();
                    if (k < probabilities.size(1) - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "Probability prediction: Not supported" << std::endl;
        }
    }
}

/**
 * @brief Demonstrate feature preprocessing effects
 */
void demonstrate_preprocessing_effects()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FEATURE PREPROCESSING EFFECTS EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Generate dataset with different feature scales
    auto [X_base, y] = generate_realistic_dataset(800, 10, 3, 0.15);

    // Create features with different scales
    auto X_unscaled = X_base.clone();
    X_unscaled.slice(1, 0, 3) *= 100.0;    // Features 0-2: large scale
    X_unscaled.slice(1, 3, 6) *= 0.01;     // Features 3-5: small scale
    // Features 6-9: original scale

    std::cout << "Original dataset statistics:" << std::endl;
    std::cout << "  Min values: " << std::get<0>(torch::min(X_unscaled, 0)) << std::endl;
    std::cout << "  Max values: " << std::get<0>(torch::max(X_unscaled, 0)) << std::endl;
    std::cout << "  Mean values: " << X_unscaled.mean(0) << std::endl;
    std::cout << "  Std values: " << X_unscaled.std(0) << std::endl;

    // Test different preprocessing approaches
    std::vector<std::pair<std::string, torch::Tensor>> preprocessing_methods = {
        {"No Preprocessing", X_unscaled},
        {"Normalization [0,1]", normalize_features(X_unscaled)},
        {"Standardization", standardize_features(X_unscaled)}
    };

    json config = { {"kernel", "rbf"}, {"C", 1.0}, {"gamma", 0.1} };

    std::cout << "\n--- Preprocessing Method Comparison ---" << std::endl;
    std::cout << std::setw(20) << "Method" << std::setw(15) << "CV Score" << std::setw(15) << "Training Time" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (const auto& [method_name, X_processed] : preprocessing_methods) {
        SVMClassifier svm(config);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto cv_scores = svm.cross_validate(X_processed, y, 5);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        double mean_cv = std::accumulate(cv_scores.begin(), cv_scores.end(), 0.0) / cv_scores.size();

        std::cout << std::setw(20) << method_name
            << std::setw(15) << std::fixed << std::setprecision(4) << mean_cv * 100 << "%"
            << std::setw(15) << duration.count() << " ms" << std::endl;
    }

    std::cout << "\nKey Insights:" << std::endl;
    std::cout << "- Normalization scales features to [0,1] range" << std::endl;
    std::cout << "- Standardization centers features at 0 with unit variance" << std::endl;
    std::cout << "- RBF kernels are particularly sensitive to feature scaling" << std::endl;
    std::cout << "- Preprocessing often improves performance significantly" << std::endl;
}

/**
 * @brief Demonstrate class imbalance handling
 */
void demonstrate_class_imbalance()
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CLASS IMBALANCE HANDLING EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create imbalanced dataset
    torch::manual_seed(42);

    // Class 0: 500 samples (majority)
    auto X0 = torch::randn({ 500, 8 }) + torch::tensor({ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
    auto y0 = torch::zeros({ 500 }, torch::kInt32);

    // Class 1: 100 samples (minority)
    auto X1 = torch::randn({ 100, 8 }) + torch::tensor({ -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 });
    auto y1 = torch::ones({ 100 }, torch::kInt32);

    // Class 2: 50 samples (very minority)
    auto X2 = torch::randn({ 50, 8 }) + torch::tensor({ 0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0 });
    auto y2 = torch::full({ 50 }, 2, torch::kInt32);

    auto X = torch::cat({ X0, X1, X2 }, 0);
    auto y = torch::cat({ y0, y1, y2 }, 0);

    // Shuffle
    auto indices = torch::randperm(X.size(0));
    X = X.index_select(0, indices);
    y = y.index_select(0, indices);

    // Standardize features
    X = standardize_features(X);

    std::cout << "Imbalanced dataset created:" << std::endl;
    std::cout << "  Class 0: 500 samples (76.9%)" << std::endl;
    std::cout << "  Class 1: 100 samples (15.4%)" << std::endl;
    std::cout << "  Class 2:  50 samples (7.7%)" << std::endl;
    std::cout << "  Total:   650 samples" << std::endl;

    // Test different strategies
    std::vector<json> strategies = {
        {{"kernel", "linear"}, {"C", 1.0}, {"multiclass_strategy", "ovr"}},
        {{"kernel", "linear"}, {"C", 10.0}, {"multiclass_strategy", "ovr"}},
        {{"kernel", "rbf"}, {"C", 1.0}, {"gamma", 0.1}, {"multiclass_strategy", "ovr"}},
        {{"kernel", "rbf"}, {"C", 10.0}, {"gamma", 0.1}, {"multiclass_strategy", "ovo"}}
    };

    std::vector<std::string> strategy_names = {
        "Linear (C=1.0, OvR)",
        "Linear (C=10.0, OvR)",
        "RBF (C=1.0, OvR)",
        "RBF (C=10.0, OvO)"
    };

    std::cout << "\n--- Strategy Comparison for Imbalanced Data ---" << std::endl;

    for (size_t i = 0; i < strategies.size(); ++i) {
        std::cout << "\n" << std::string(30, '-') << std::endl;
        std::cout << "Strategy: " << strategy_names[i] << std::endl;

        SVMClassifier svm(strategies[i]);
        svm.fit(X, y);

        auto metrics = svm.evaluate(X, y);
        print_evaluation_metrics(metrics, strategy_names[i] + " Performance");

        // Per-class analysis
        std::cout << "\nPer-class analysis:" << std::endl;
        for (int class_idx = 0; class_idx < 3; ++class_idx) {
            int tp = metrics.confusion_matrix[class_idx][class_idx];
            int total = 0;
            for (int j = 0; j < 3; ++j) {
                total += metrics.confusion_matrix[class_idx][j];
            }
            double class_recall = (total > 0) ? static_cast<double>(tp) / total : 0.0;
            std::cout << "  Class " << class_idx << " recall: "
                << std::fixed << std::setprecision(4) << class_recall * 100 << "%" << std::endl;
        }
    }

    std::cout << "\nRecommendations for imbalanced data:" << std::endl;
    std::cout << "- Increase C parameter to give more weight to training errors" << std::endl;
    std::cout << "- Consider One-vs-One strategy for better minority class handling" << std::endl;
    std::cout << "- Use class-specific evaluation metrics (precision, recall per class)" << std::endl;
    std::cout << "- Consider resampling techniques in preprocessing" << std::endl;
}

int main()
{
    try {
        std::cout << "Advanced SVM Classifier Usage Examples" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Set single-threaded mode for reproducible results
        torch::set_num_threads(1);

        // Run comprehensive examples
        demonstrate_hyperparameter_tuning();
        demonstrate_model_evaluation();
        demonstrate_preprocessing_effects();
        demonstrate_class_imbalance();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ALL ADVANCED EXAMPLES COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "\nKey Takeaways:" << std::endl;
        std::cout << "1. Hyperparameter tuning is crucial for optimal performance" << std::endl;
        std::cout << "2. Feature preprocessing significantly affects RBF and polynomial kernels" << std::endl;
        std::cout << "3. Cross-validation provides robust performance estimates" << std::endl;
        std::cout << "4. Different kernels and strategies work better for different data types" << std::endl;
        std::cout << "5. Class imbalance requires special consideration in model selection" << std::endl;
        std::cout << "6. Linear kernels are fastest and work well for high-dimensional data" << std::endl;
        std::cout << "7. RBF kernels provide good general-purpose non-linear classification" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}