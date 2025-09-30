#include <iostream>

#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <numeric>
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>

using namespace svm_classifier;
using json = nlohmann::json;

/**
 * @brief Generate a more realistic multi-class dataset with noise
 */
std::pair<torch::Tensor, torch::Tensor> generate_realistic_dataset(
    int n_samples,
    int n_features,
    int n_classes,
    double noise_factor = 0.1) {
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
torch::Tensor normalize_features(const torch::Tensor& X) {
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
torch::Tensor standardize_features(const torch::Tensor& X) {
    auto mean = X.mean(0);
    auto std = X.std(0);

    // Avoid division by zero
    std = torch::where(std == 0.0, torch::ones_like(std), std);

    return (X - mean) / std;
}

/**
 * @brief Print detailed evaluation metrics
 */
void print_evaluation_metrics(const EvaluationMetrics& metrics, const std::string& title) {
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
 * @brief Demonstrate manual hyperparameter tuning
 */
void demonstrate_hyperparameter_tuning() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HYPERPARAMETER COMPARISON EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    auto [X_full, y_full] = generate_realistic_dataset(800, 10, 3, 0.3);
    X_full = standardize_features(X_full);

    int n_train = 600;
    auto X_train = X_full.slice(0, 0, n_train);
    auto y_train = y_full.slice(0, 0, n_train);
    auto X_test = X_full.slice(0, n_train);
    auto y_test = y_full.slice(0, n_train);

    std::cout << "Dataset: " << X_train.size(0) << " train, " << X_test.size(0) << " test samples, "
              << X_train.size(1) << " features" << std::endl;

    std::vector<double> c_values = { 0.1, 1.0, 10.0, 100.0 };

    std::cout << "\n--- Linear SVM C Parameter Comparison ---" << std::endl;
    std::cout << std::setw(10) << "C" << std::setw(15) << "Accuracy" << std::endl;
    std::cout << std::string(25, '-') << std::endl;

    for (double c : c_values) {
        json config = { { "kernel", "linear" }, { "C", c } };
        SVMClassifier svm(config);
        svm.fit(X_train, y_train);
        double accuracy = svm.score(X_test, y_test);

        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << c << std::setw(15)
                  << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;
    }

    std::vector<double> gamma_values = { 0.01, 0.1, 1.0 };

    std::cout << "\n--- RBF SVM Gamma Parameter Comparison (C=10.0) ---" << std::endl;
    std::cout << std::setw(10) << "Gamma" << std::setw(15) << "Accuracy" << std::endl;
    std::cout << std::string(25, '-') << std::endl;

    for (double gamma : gamma_values) {
        json config = { { "kernel", "rbf" }, { "C", 10.0 }, { "gamma", gamma } };
        SVMClassifier svm(config);
        svm.fit(X_train, y_train);
        double accuracy = svm.score(X_test, y_test);

        std::cout << std::setw(10) << std::fixed << std::setprecision(2) << gamma << std::setw(15)
                  << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;
    }
}

/**
 * @brief Demonstrate model evaluation and validation
 */
void demonstrate_model_evaluation() {
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
    std::cout << "  Classes:  " << std::get<0>(at::_unique(y_train)).size(0) << std::endl;

    // Configure different models for comparison
    std::vector<json> model_configs = {
        { { "kernel", "linear" }, { "C", 1.0 }, { "multiclass_strategy", "ovr" } },
        { { "kernel", "linear" }, { "C", 1.0 }, { "multiclass_strategy", "ovo" } },
        { { "kernel", "rbf" }, { "C", 10.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovr" } },
        { { "kernel", "rbf" }, { "C", 10.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovo" } },
        { { "kernel", "polynomial" }, { "degree", 3 }, { "C", 1.0 } }
    };

    std::vector<std::string> model_names = { "Linear (OvR)", "Linear (OvO)", "RBF (OvR)",
                                             "RBF (OvO)", "Polynomial" };

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
        auto training_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Training time: " << training_duration.count() << " ms" << std::endl;

        // Evaluate on training set
        auto train_metrics = svm.evaluate(X_train, y_train);
        print_evaluation_metrics(train_metrics, "Training Set Performance");

        // Evaluate on test set
        auto test_metrics = svm.evaluate(X_test, y_test);
        print_evaluation_metrics(test_metrics, "Test Set Performance");

        // Prediction summary
        auto predictions = svm.predict(X_test);
        std::cout << "\nPredictions completed for " << predictions.size(0) << " test samples"
                  << std::endl;
    }
}

/**
 * @brief Demonstrate feature preprocessing effects
 */
void demonstrate_preprocessing_effects() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FEATURE PREPROCESSING EFFECTS EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Generate dataset with different feature scales
    auto [X_base, y] = generate_realistic_dataset(800, 10, 3, 0.15);

    // Create features with different scales
    auto X_unscaled = X_base.clone();
    X_unscaled.slice(1, 0, 3) *= 100.0; // Features 0-2: large scale
    X_unscaled.slice(1, 3, 6) *= 0.01;  // Features 3-5: small scale
    // Features 6-9: original scale

    std::cout << "Original dataset statistics:" << std::endl;
    std::cout << "  Min values: " << std::get<0>(torch::min(X_unscaled, 0)) << std::endl;
    std::cout << "  Max values: " << std::get<0>(torch::max(X_unscaled, 0)) << std::endl;
    std::cout << "  Mean values: " << X_unscaled.mean(0) << std::endl;
    std::cout << "  Std values: " << X_unscaled.std(0) << std::endl;

    // Test different preprocessing approaches
    std::vector<std::pair<std::string, torch::Tensor>> preprocessing_methods = {
        { "No Preprocessing", X_unscaled },
        { "Normalization [0,1]", normalize_features(X_unscaled) },
        { "Standardization", standardize_features(X_unscaled) }
    };

    json config = { { "kernel", "rbf" }, { "C", 1.0 }, { "gamma", 0.1 } };

    // Split data for testing
    int n_train = 600;
    auto X_test_slice = X_unscaled.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    std::cout << "\n--- Preprocessing Method Comparison ---" << std::endl;
    std::cout << std::setw(20) << "Method" << std::setw(15) << "Test Accuracy" << std::endl;
    std::cout << std::string(35, '-') << std::endl;

    for (const auto& [method_name, X_processed] : preprocessing_methods) {
        auto X_train_slice = X_processed.slice(0, 0, n_train);
        auto y_train_slice = y.slice(0, 0, n_train);
        auto X_test_proc = (method_name == "No Preprocessing") ? X_test_slice
                           : (method_name == "Normalization [0,1]")
                               ? normalize_features(X_test_slice)
                               : standardize_features(X_test_slice);

        SVMClassifier svm(config);
        svm.fit(X_train_slice, y_train_slice);
        double accuracy = svm.score(X_test_proc, y_test);

        std::cout << std::setw(20) << method_name << std::setw(15) << std::fixed
                  << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;
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
void demonstrate_class_imbalance() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CLASS IMBALANCE HANDLING EXAMPLE" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create imbalanced dataset
    torch::manual_seed(42);

    // Class 0: 500 samples (majority)
    auto X0 = torch::randn({ 500, 8 }) + torch::tensor({ 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 });
    auto y0 = torch::zeros({ 500 }, torch::kInt32);

    // Class 1: 100 samples (minority)
    auto X1 =
        torch::randn({ 100, 8 }) + torch::tensor({ -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 });
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
        { { "kernel", "linear" }, { "C", 1.0 }, { "multiclass_strategy", "ovr" } },
        { { "kernel", "linear" }, { "C", 10.0 }, { "multiclass_strategy", "ovr" } },
        { { "kernel", "rbf" }, { "C", 1.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovr" } },
        { { "kernel", "rbf" }, { "C", 10.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovo" } }
    };

    std::vector<std::string> strategy_names = { "Linear (C=1.0, OvR)", "Linear (C=10.0, OvR)",
                                                "RBF (C=1.0, OvR)", "RBF (C=10.0, OvO)" };

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
            std::cout << "  Class " << class_idx << " recall: " << std::fixed
                      << std::setprecision(4) << class_recall * 100 << "%" << std::endl;
        }
    }

    std::cout << "\nRecommendations for imbalanced data:" << std::endl;
    std::cout << "- Increase C parameter to give more weight to training errors" << std::endl;
    std::cout << "- Consider One-vs-One strategy for better minority class handling" << std::endl;
    std::cout << "- Use class-specific evaluation metrics (precision, recall per class)"
              << std::endl;
    std::cout << "- Consider resampling techniques in preprocessing" << std::endl;
}

int main() {
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
        std::cout << "2. Feature preprocessing significantly affects RBF and polynomial kernels"
                  << std::endl;
        std::cout << "3. Cross-validation provides robust performance estimates" << std::endl;
        std::cout << "4. Different kernels and strategies work better for different data types"
                  << std::endl;
        std::cout << "5. Class imbalance requires special consideration in model selection"
                  << std::endl;
        std::cout << "6. Linear kernels are fastest and work well for high-dimensional data"
                  << std::endl;
        std::cout << "7. RBF kernels provide good general-purpose non-linear classification"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}