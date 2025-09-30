#pragma once

#include <stdexcept>

#include <memory>
#include <string>
#include <vector>

namespace svm_classifier {

/**
 * @brief Supported kernel types
 */
enum class KernelType {
    LINEAR,     ///< Linear kernel: <x, y>
    RBF,        ///< Radial Basis Function: exp(-gamma * ||x - y||^2)
    POLYNOMIAL, ///< Polynomial: (gamma * <x, y> + coef0)^degree
    SIGMOID     ///< Sigmoid: tanh(gamma * <x, y> + coef0)
};

/**
 * @brief Multiclass classification strategies
 */
enum class MulticlassStrategy {
    ONE_VS_REST, ///< One-vs-Rest (OvR) strategy
    ONE_VS_ONE   ///< One-vs-One (OvO) strategy
};

/**
 * @brief SVM library type selection
 */
enum class SVMLibrary {
    LIBLINEAR, ///< Use liblinear (for linear kernels)
    LIBSVM     ///< Use libsvm (for non-linear kernels)
};

/**
 * @brief Training result status
 */
enum class TrainingStatus {
    SUCCESS,
    INVALID_PARAMETERS,
    INSUFFICIENT_DATA,
    MEMORY_ERROR,
    CONVERGENCE_ERROR
};

/**
 * @brief Prediction result structure
 */
struct PredictionResult {
    std::vector<int> predictions;                     ///< Predicted class labels
    std::vector<std::vector<double>> probabilities;   ///< Class probabilities (if available)
    std::vector<std::vector<double>> decision_values; ///< Decision function values
    bool has_probabilities = false;                   ///< Whether probabilities are available
};

/**
 * @brief Training metrics structure
 */
struct TrainingMetrics {
    double training_time = 0.0;   ///< Training time in seconds
    int support_vectors = 0;      ///< Number of support vectors
    int iterations = 0;           ///< Number of iterations
    double objective_value = 0.0; ///< Final objective value
    TrainingStatus status = TrainingStatus::SUCCESS;
};

/**
 * @brief Model evaluation metrics
 */
struct EvaluationMetrics {
    double accuracy = 0.0;                          ///< Classification accuracy
    double precision = 0.0;                         ///< Macro-averaged precision
    double recall = 0.0;                            ///< Macro-averaged recall
    double f1_score = 0.0;                          ///< Macro-averaged F1-score
    std::vector<std::vector<int>> confusion_matrix; ///< Confusion matrix
};

/**
 * @brief Convert kernel type to string
 */
inline std::string kernel_type_to_string(KernelType kernel) {
    switch (kernel) {
    case KernelType::LINEAR:
        return "linear";
    case KernelType::RBF:
        return "rbf";
    case KernelType::POLYNOMIAL:
        return "polynomial";
    case KernelType::SIGMOID:
        return "sigmoid";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert string to kernel type
 */
inline KernelType string_to_kernel_type(const std::string& kernel_str) {
    if (kernel_str == "linear")
        return KernelType::LINEAR;
    if (kernel_str == "rbf")
        return KernelType::RBF;
    if (kernel_str == "polynomial" || kernel_str == "poly")
        return KernelType::POLYNOMIAL;
    if (kernel_str == "sigmoid")
        return KernelType::SIGMOID;
    throw std::invalid_argument("Unknown kernel type: " + kernel_str);
}

/**
 * @brief Convert multiclass strategy to string
 */
inline std::string multiclass_strategy_to_string(MulticlassStrategy strategy) {
    switch (strategy) {
    case MulticlassStrategy::ONE_VS_REST:
        return "ovr";
    case MulticlassStrategy::ONE_VS_ONE:
        return "ovo";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert string to multiclass strategy
 */
inline MulticlassStrategy string_to_multiclass_strategy(const std::string& strategy_str) {
    if (strategy_str == "ovr" || strategy_str == "one_vs_rest") {
        return MulticlassStrategy::ONE_VS_REST;
    }
    if (strategy_str == "ovo" || strategy_str == "one_vs_one") {
        return MulticlassStrategy::ONE_VS_ONE;
    }
    throw std::invalid_argument("Unknown multiclass strategy: " + strategy_str);
}

/**
 * @brief Determine which SVM library to use based on kernel type
 */
inline SVMLibrary get_svm_library(KernelType kernel) {
    return (kernel == KernelType::LINEAR) ? SVMLibrary::LIBLINEAR : SVMLibrary::LIBSVM;
}

} // namespace svm_classifier