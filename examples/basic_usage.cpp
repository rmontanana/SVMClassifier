#include <iostream>

#include <nlohmann/json.hpp>
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>

using namespace svm_classifier;
using json = nlohmann::json;

/**
 * @brief Generate synthetic 2D classification dataset
 */
std::pair<torch::Tensor, torch::Tensor> generate_classification_data(
    int n_samples,
    int n_classes = 3) {
    torch::manual_seed(42);

    auto X = torch::randn({ n_samples, 2 });
    auto y = torch::zeros({ n_samples }, torch::kInt);

    for (int i = 0; i < n_samples; ++i) {
        double x_val = X[i][0].item<double>();
        double y_val = X[i][1].item<double>();

        if (x_val > 0.5 && y_val > 0.5) {
            y[i] = 0;
        } else if (x_val <= 0.5 && y_val > 0.5) {
            y[i] = 1;
        } else {
            y[i] = 2;
        }
    }

    X += torch::randn_like(X) * 0.1;
    return { X, y };
}

/**
 * @brief Demonstrate basic SVM usage
 */
void basic_svm_example() {
    std::cout << "=== Basic SVM Classification Example ===" << std::endl;

    // Generate synthetic data
    auto [X, y] = generate_classification_data(200, 3);

    // Split into train/test sets (80/20 split)
    int n_train = 160;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    std::cout << "Dataset: " << X_train.size(0) << " training, " << X_test.size(0)
              << " test samples" << std::endl;

    // Create SVM classifier with default parameters
    SVMClassifier svm;

    std::cout << "Training SVM..." << std::endl;
    svm.fit(X_train, y_train);

    double accuracy = svm.score(X_test, y_test);
    std::cout << "Test accuracy: " << (accuracy * 100.0) << "%" << std::endl << std::endl;
}

/**
 * @brief Demonstrate different kernels
 */
void kernel_comparison_example() {
    std::cout << "=== Kernel Comparison Example ===" << std::endl;

    auto [X, y] = generate_classification_data(300, 2);
    int n_train = 240;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    std::vector<KernelType> kernels = { KernelType::LINEAR, KernelType::RBF,
                                        KernelType::POLYNOMIAL };

    for (auto kernel : kernels) {
        SVMClassifier svm(kernel, 1.0, MulticlassStrategy::ONE_VS_REST);
        svm.fit(X_train, y_train);
        double accuracy = svm.score(X_test, y_test);

        std::cout << kernel_type_to_string(kernel) << " kernel: " << (accuracy * 100.0) << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Demonstrate JSON parameter configuration
 */
void json_configuration_example() {
    std::cout << "=== JSON Configuration Example ===" << std::endl;

    json config = {
        { "kernel", "rbf" }, { "C", 10.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovo" }
    };

    std::cout << "Config: " << config.dump() << std::endl;

    auto [X, y] = generate_classification_data(200, 3);
    int n_train = 160;
    auto X_train = X.slice(0, 0, n_train);
    auto y_train = y.slice(0, 0, n_train);
    auto X_test = X.slice(0, n_train);
    auto y_test = y.slice(0, n_train);

    SVMClassifier svm(config);
    svm.fit(X_train, y_train);

    double accuracy = svm.score(X_test, y_test);
    std::cout << "Accuracy: " << (accuracy * 100.0) << "%" << std::endl << std::endl;
}

int main() {
    try {
        std::cout << "SVM Classifier - Basic Examples" << std::endl;
        std::cout << "================================" << std::endl << std::endl;

        torch::set_num_threads(1);

        basic_svm_example();
        kernel_comparison_example();
        json_configuration_example();

        std::cout << "Examples completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}