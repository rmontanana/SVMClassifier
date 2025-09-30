#pragma once

#include "data_converter.hpp"
#include "kernel_parameters.hpp"
#include "multiclass_strategy.hpp"
#include "types.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <torch/torch.h>

namespace svm_classifier {

/**
 * @brief Support Vector Machine Classifier with scikit-learn compatible API
 *
 * This class provides a unified interface for SVM classification using both
 * liblinear (for linear kernels) and libsvm (for non-linear kernels).
 * It supports multiclass classification through One-vs-Rest and One-vs-One strategies.
 */
class SVMClassifier {
public:
    /**
     * @brief Default constructor with default parameters
     */
    SVMClassifier();

    /**
     * @brief Constructor with JSON parameters
     * @param config JSON configuration object
     */
    explicit SVMClassifier(const nlohmann::json& config);

    /**
     * @brief Constructor with explicit parameters
     * @param kernel Kernel type
     * @param C Regularization parameter
     * @param multiclass_strategy Multiclass strategy
     */
    SVMClassifier(
        KernelType kernel,
        double C = 1.0,
        MulticlassStrategy multiclass_strategy = MulticlassStrategy::ONE_VS_REST);

    /**
     * @brief Destructor
     */
    ~SVMClassifier();

    /**
     * @brief Copy constructor (deleted - models are not copyable)
     */
    SVMClassifier(const SVMClassifier&) = delete;

    /**
     * @brief Copy assignment (deleted - models are not copyable)
     */
    SVMClassifier& operator=(const SVMClassifier&) = delete;

    /**
     * @brief Move constructor
     */
    SVMClassifier(SVMClassifier&&) noexcept;

    /**
     * @brief Move assignment
     */
    SVMClassifier& operator=(SVMClassifier&&) noexcept;

    /**
     * @brief Train the SVM classifier
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param y Target tensor of shape (n_samples,) with class labels
     * @return Training metrics
     * @throws std::invalid_argument if input data is invalid
     * @throws std::runtime_error if training fails
     */
    TrainingMetrics fit(const torch::Tensor& X, const torch::Tensor& y);

    /**
     * @brief Predict class labels for samples
     * @param X Feature tensor of shape (n_samples, n_features)
     * @return Tensor of predicted class labels
     * @throws std::runtime_error if model is not fitted
     */
    torch::Tensor predict(const torch::Tensor& X);

    /**
     * @brief Predict class probabilities for samples
     * @param X Feature tensor of shape (n_samples, n_features)
     * @return Tensor of shape (n_samples, n_classes) with class probabilities
     * @throws std::runtime_error if model is not fitted or doesn't support probabilities
     */
    torch::Tensor predict_proba(const torch::Tensor& X);

    /**
     * @brief Get decision function values
     * @param X Feature tensor of shape (n_samples, n_features)
     * @return Tensor with decision function values
     * @throws std::runtime_error if model is not fitted
     */
    torch::Tensor decision_function(const torch::Tensor& X);

    /**
     * @brief Calculate accuracy score on test data
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param y_true True labels tensor of shape (n_samples,)
     * @return Accuracy score (fraction of correctly predicted samples)
     * @throws std::runtime_error if model is not fitted
     */
    double score(const torch::Tensor& X, const torch::Tensor& y_true);

    /**
     * @brief Calculate detailed evaluation metrics
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param y_true True labels tensor of shape (n_samples,)
     * @return Evaluation metrics including precision, recall, F1-score
     */
    EvaluationMetrics evaluate(const torch::Tensor& X, const torch::Tensor& y_true);

    /**
     * @brief Set parameters from JSON configuration
     * @param config JSON configuration object
     * @throws std::invalid_argument if parameters are invalid
     */
    void set_parameters(const nlohmann::json& config);

    /**
     * @brief Get current parameters as JSON
     * @return JSON object with current parameters
     */
    nlohmann::json get_parameters() const;

    /**
     * @brief Check if the model is fitted/trained
     * @return True if model is fitted
     */
    bool is_fitted() const {
        return is_fitted_;
    }

    /**
     * @brief Get the number of classes
     * @return Number of classes (0 if not fitted)
     */
    int get_n_classes() const;

    /**
     * @brief Get unique class labels
     * @return Vector of unique class labels
     */
    std::vector<int> get_classes() const;

    /**
     * @brief Get the number of features
     * @return Number of features (0 if not fitted)
     */
    int get_n_features() const {
        return n_features_;
    }

    /**
     * @brief Get training metrics from last fit
     * @return Training metrics
     */
    TrainingMetrics get_training_metrics() const {
        return training_metrics_;
    }

    /**
     * @brief Check if the current model supports probability prediction
     * @return True if probabilities are supported
     */
    bool supports_probability() const;

    /**
     * @brief Save model to file
     * @param filename Path to save the model
     * @throws std::runtime_error if saving fails
     */
    void save_model(const std::string& filename) const;

    /**
     * @brief Load model from file
     * @param filename Path to load the model from
     * @throws std::runtime_error if loading fails
     */
    void load_model(const std::string& filename);

    /**
     * @brief Get kernel type
     * @return Current kernel type
     */
    KernelType get_kernel_type() const {
        return params_.get_kernel_type();
    }

    /**
     * @brief Get multiclass strategy
     * @return Current multiclass strategy
     */
    MulticlassStrategy get_multiclass_strategy() const {
        return params_.get_multiclass_strategy();
    }

    /**
     * @brief Get SVM library being used
     * @return SVM library type
     */
    SVMLibrary get_svm_library() const {
        return ::svm_classifier::get_svm_library(params_.get_kernel_type());
    }

    /**
     * @brief Perform cross-validation
     * @param X Feature tensor
     * @param y Target tensor
     * @param cv Number of folds (default: 5)
     * @return Cross-validation scores for each fold
     */
    std::vector<double> cross_validate(const torch::Tensor& X, const torch::Tensor& y, int cv = 5);

    /**
     * @brief Find optimal hyperparameters using grid search
     * @param X Feature tensor
     * @param y Target tensor
     * @param param_grid JSON object with parameter grid
     * @param cv Number of cross-validation folds
     * @return JSON object with best parameters and score
     */
    nlohmann::json grid_search(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const nlohmann::json& param_grid,
        int cv = 5);

    /**
     * @brief Get feature importance (for linear kernels only)
     * @return Tensor with feature weights/importance
     * @throws std::runtime_error if not supported for current kernel
     */
    torch::Tensor get_feature_importance() const;

    /**
     * @brief Reset the classifier (clear trained model)
     */
    void reset();

private:
    KernelParameters params_;                                     ///< Model parameters
    std::unique_ptr<MulticlassStrategyBase> multiclass_strategy_; ///< Multiclass strategy
    std::unique_ptr<DataConverter> data_converter_;               ///< Data converter

    bool is_fitted_;                   ///< Whether model is fitted
    int n_features_;                   ///< Number of features
    TrainingMetrics training_metrics_; ///< Last training metrics

    /**
     * @brief Validate input data
     * @param X Feature tensor
     * @param y Target tensor (optional)
     * @param check_fitted Whether to check if model is fitted
     */
    void validate_input(
        const torch::Tensor& X,
        const torch::Tensor& y = torch::Tensor(),
        bool check_fitted = false);

    /**
     * @brief Initialize multiclass strategy based on current parameters
     */
    void initialize_multiclass_strategy();

    /**
     * @brief Calculate confusion matrix
     * @param y_true True labels
     * @param y_pred Predicted labels
     * @return Confusion matrix
     */
    std::vector<std::vector<int>> calculate_confusion_matrix(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred);

    /**
     * @brief Calculate precision, recall, and F1-score from confusion matrix
     * @param confusion_matrix Confusion matrix
     * @return Tuple of (precision, recall, f1_score)
     */
    std::tuple<double, double, double> calculate_metrics_from_confusion_matrix(
        const std::vector<std::vector<int>>& confusion_matrix);

    /**
     * @brief Split data for cross-validation
     * @param X Feature tensor
     * @param y Target tensor
     * @param fold Current fold
     * @param n_folds Total number of folds
     * @return Tuple of (X_train, y_train, X_val, y_val)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    split_for_cv(const torch::Tensor& X, const torch::Tensor& y, int fold, int n_folds);

    /**
     * @brief Generate parameter combinations for grid search
     * @param param_grid JSON parameter grid
     * @return Vector of parameter combinations
     */
    std::vector<nlohmann::json> generate_param_combinations(const nlohmann::json& param_grid);
};

} // namespace svm_classifier