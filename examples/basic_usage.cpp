#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace svm_classifier;
using json = nlohmann::json;

/**
 * @brief Generate synthetic 2D classification dataset
 * @param n_samples Number of samples to generate
 * @param n_classes Number of classes
 * @return Pair of (features, labels)
 */
std::pair<torch::Tensor, torch::Tensor> generate_classification_data(int n_samples, int n_classes = 3)
{
    torch::manual_seed(42);  // For reproducibility

    // Generate random features
    auto X = torch::randn({ n_samples, 2 });

    // Create clusters for different classes
    auto y = torch::zeros({ n_samples }, torch::kInt);

    for (int i = 0; i < n_samples; ++i) {
        // Simple clustering based on position
        double x_val = X[i][0].item<double>();
        double y_val = X[i][1].item<double>();

        if (x_val > 0.5 && y_val > 0.5) {
            y[i] = 0;  // Class 0: top-right
        } else if (x_val <= 0.5 && y_val > 0.5) {
            y[i] = 1;  // Class 1: top-left
        } else {
            y[i] = 2;  // Class 2: bottom
        }
    }

    // Add some noise to make it more interesting
    X += torch::randn_like(X) * 0.1;

    return { X, y };
}

/**
 * @brief Print tensor statistics
 */
void print_tensor_stats(const torch::Tensor& tensor, const std::string& name)
{
    std::cout << name << " shape: [" << tensor.size(0) << ", " << tensor.size(1) << "]" << std::endl;
    std::cout << name << " min: " << tensor.min().item<double>() << std::endl;
    std::cout << name << " max: " << tensor.max().item<double>() << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Demonstrate basic SVM usage
 */
void basic_svm_example()
{
    std::cout << "=== Basic SVM Classification Example ===" << std::endl;

    // Generate synthetic data
    auto [X, y] = generate_classification_data(200, 3);

    // Split into train/test sets (80/20 split)
    int n_train = 160;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    std::cout << "Dataset created:" << std::endl;
    print_tensor_stats(X_train, "X_train");
    std::cout << "Unique classes in y_train: ";
    auto unique_classes = torch::unique(y_train);
    for (int i = 0; i < unique_classes.size(0); ++i) {
        std::cout << unique_classes[i].item<int>() << " ";
    }
    std::cout << std::endl << std::endl;

    // Create SVM classifier with default parameters
    SVMClassifier svm;

    // Train the model
    std::cout << "Training SVM with default parameters..." << std::endl;
    auto training_metrics = svm.fit(X_train, y_train);

    std::cout << "Training completed:" << std::endl;
    std::cout << "  Training time: " << training_metrics.training_time << " seconds" << std::endl;
    std::cout << "  Support vectors: " << training_metrics.support_vectors << std::endl;
    std::cout << "  Status: " << (training_metrics.status == TrainingStatus::SUCCESS ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // Make predictions
    std::cout << "Making predictions..." << std::endl;
    auto predictions = svm.predict(X_test);

    // Calculate accuracy
    double accuracy = svm.score(X_test, y_test);
    std::cout << "Test accuracy: " << (accuracy * 100.0) << "%" << std::endl;

    // Get detailed evaluation metrics
    auto eval_metrics = svm.evaluate(X_test, y_test);
    std::cout << "Detailed metrics:" << std::endl;
    std::cout << "  Precision: " << (eval_metrics.precision * 100.0) << "%" << std::endl;
    std::cout << "  Recall: " << (eval_metrics.recall * 100.0) << "%" << std::endl;
    std::cout << "  F1-score: " << (eval_metrics.f1_score * 100.0) << "%" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Demonstrate different kernels
 */
void kernel_comparison_example()
{
    std::cout << "=== Kernel Comparison Example ===" << std::endl;

    // Generate more complex dataset
    auto [X, y] = generate_classification_data(300, 2);

    // Split data
    int n_train = 240;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    // Test different kernels
    std::vector<KernelType> kernels = {
        KernelType::LINEAR,
        KernelType::RBF,
        KernelType::POLYNOMIAL,
        KernelType::SIGMOID
    };

    for (auto kernel : kernels) {
        std::cout << "Testing " << kernel_type_to_string(kernel) << " kernel:" << std::endl;

        // Create classifier with specific kernel
        SVMClassifier svm(kernel, 1.0, MulticlassStrategy::ONE_VS_REST);

        // Train and evaluate
        auto training_metrics = svm.fit(X_train, y_train);
        double accuracy = svm.score(X_test, y_test);

        std::cout << "  Training time: " << training_metrics.training_time << " seconds" << std::endl;
        std::cout << "  Test accuracy: " << (accuracy * 100.0) << "%" << std::endl;
        std::cout << "  Library used: " << (svm.get_svm_library() == SVMLibrary::LIBLINEAR ? "liblinear" : "libsvm") << std::endl;
        std::cout << std::endl;
    }
}

/**
 * @brief Demonstrate JSON parameter configuration
 */
void json_configuration_example()
{
    std::cout << "=== JSON Configuration Example ===" << std::endl;

    // Create JSON configuration
    json config = {
        {"kernel", "rbf"},
        {"C", 10.0},
        {"gamma", 0.1},
        {"multiclass_strategy", "ovo"},
        {"probability", true},
        {"tolerance", 1e-4}
    };

    std::cout << "Configuration JSON:" << std::endl;
    std::cout << config.dump(2) << std::endl << std::endl;

    // Generate data
    auto [X, y] = generate_classification_data(200, 3);
    int n_train = 160;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    // Create classifier from JSON
    SVMClassifier svm(config);

    // Train the model
    auto training_metrics = svm.fit(X_train, y_train);

    // Make predictions with probabilities
    auto predictions = svm.predict(X_test);

    if (svm.supports_probability()) {
        auto probabilities = svm.predict_proba(X_test);
        std::cout << "Probability predictions shape: [" << probabilities.size(0) << ", " << probabilities.size(1) << "]" << std::endl;
    }

    double accuracy = svm.score(X_test, y_test);
    std::cout << "Final accuracy: " << (accuracy * 100.0) << "%" << std::endl;

    // Show current parameters
    auto current_params = svm.get_parameters();
    std::cout << "Current parameters:" << std::endl;
    std::cout << current_params.dump(2) << std::endl;
}

/**
 * @brief Demonstrate cross-validation
 */
void cross_validation_example()
{
    std::cout << "=== Cross-Validation Example ===" << std::endl;

    // Generate dataset
    auto [X, y] = generate_classification_data(500, 3);

    // Create SVM classifier
    SVMClassifier svm(KernelType::RBF, 1.0);

    // Perform 5-fold cross-validation
    std::cout << "Performing 5-fold cross-validation..." << std::endl;
    auto cv_scores = svm.cross_validate(X, y, 5);

    std::cout << "Cross-validation scores:" << std::endl;
    double mean_score = 0.0;
    for (size_t i = 0; i < cv_scores.size(); ++i) {
        std::cout << "  Fold " << (i + 1) << ": " << (cv_scores[i] * 100.0) << "%" << std::endl;
        mean_score += cv_scores[i];
    }
    mean_score /= cv_scores.size();

    std::cout << "Mean CV score: " << (mean_score * 100.0) << "%" << std::endl;

    // Calculate standard deviation
    double std_dev = 0.0;
    for (auto score : cv_scores) {
        std_dev += (score - mean_score) * (score - mean_score);
    }
    std_dev = std::sqrt(std_dev / cv_scores.size());

    std::cout << "Standard deviation: " << (std_dev * 100.0) << "%" << std::endl;
}

int main()
{
    try {
        std::cout << "SVM Classifier Examples" << std::endl;
        std::cout << "======================" << std::endl << std::endl;

        // Set PyTorch to single-threaded for reproducible results
        torch::set_num_threads(1);

        // Run examples
        basic_svm_example();
        std::cout << std::string(50, '-') << std::endl;

        kernel_comparison_example();
        std::cout << std::string(50, '-') << std::endl;

        json_configuration_example();
        std::cout << std::string(50, '-') << std::endl;

        cross_validation_example();

        std::cout << std::endl << "All examples completed successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}