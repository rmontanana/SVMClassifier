#pragma once

#include "data_converter.hpp"
#include "kernel_parameters.hpp"
#include "types.hpp"

#include <memory>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

// Forward declarations for external library structures
struct svm_model;
struct model;

namespace svm_classifier {

/**
 * @brief Abstract base class for multiclass classification strategies
 */
class MulticlassStrategyBase {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~MulticlassStrategyBase() = default;

    /**
     * @brief Train the multiclass classifier
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param y Target tensor of shape (n_samples,)
     * @param params Kernel parameters
     * @param converter Data converter instance
     * @return Training metrics
     */
    virtual TrainingMetrics
    fit(const torch::Tensor& X, const torch::Tensor& y, const KernelParameters& params, DataConverter& converter) = 0;

    /**
     * @brief Predict class labels
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param converter Data converter instance
     * @return Predicted class labels
     */
    virtual std::vector<int> predict(const torch::Tensor& X, DataConverter& converter) = 0;

    /**
     * @brief Predict class probabilities
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param converter Data converter instance
     * @return Class probabilities for each sample
     */
    virtual std::vector<std::vector<double>> predict_proba(const torch::Tensor& X, DataConverter& converter) = 0;

    /**
     * @brief Get decision function values
     * @param X Feature tensor of shape (n_samples, n_features)
     * @param converter Data converter instance
     * @return Decision function values
     */
    virtual std::vector<std::vector<double>> decision_function(const torch::Tensor& X, DataConverter& converter) = 0;

    /**
     * @brief Get unique class labels
     * @return Vector of unique class labels
     */
    virtual std::vector<int> get_classes() const = 0;

    /**
     * @brief Check if the model supports probability prediction
     * @return True if probabilities are supported
     */
    virtual bool supports_probability() const = 0;

    /**
     * @brief Get number of classes
     * @return Number of classes
     */
    virtual int get_n_classes() const = 0;

    /**
     * @brief Get strategy type
     * @return Multiclass strategy type
     */
    virtual MulticlassStrategy get_strategy_type() const = 0;

protected:
    std::vector<int> classes_; ///< Unique class labels
    bool is_trained_ = false;  ///< Whether the model is trained
};

/**
 * @brief One-vs-Rest (OvR) multiclass strategy
 */
class OneVsRestStrategy : public MulticlassStrategyBase {
public:
    /**
     * @brief Constructor
     */
    OneVsRestStrategy();

    /**
     * @brief Destructor
     */
    ~OneVsRestStrategy() override;

    TrainingMetrics fit(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const KernelParameters& params,
        DataConverter& converter) override;

    std::vector<int> predict(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<std::vector<double>> predict_proba(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<std::vector<double>> decision_function(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<int> get_classes() const override {
        return classes_;
    }

    bool supports_probability() const override;

    int get_n_classes() const override {
        return static_cast<int>(classes_.size());
    }

    MulticlassStrategy get_strategy_type() const override {
        return MulticlassStrategy::ONE_VS_REST;
    }

private:
    std::vector<std::unique_ptr<svm_model>> svm_models_; ///< SVM models (one per class)
    std::vector<std::unique_ptr<model>> linear_models_;  ///< Linear models (one per class)
    std::vector<std::unique_ptr<DataConverter>>
        data_converters_;     ///< Data converters (one per class) to keep training data alive
    KernelParameters params_; ///< Stored parameters
    SVMLibrary library_type_; ///< Which library is being used

    /**
     * @brief Create binary labels for one-vs-rest
     * @param y Original labels
     * @param positive_class Positive class label
     * @return Binary labels (+1 for positive class, -1 for others)
     */
    torch::Tensor create_binary_labels(const torch::Tensor& y, int positive_class);

    /**
     * @brief Train a single binary classifier
     * @param X Feature tensor
     * @param y_binary Binary labels
     * @param params Kernel parameters
     * @param converter Data converter
     * @param class_idx Index of the class being trained
     * @return Training time for this classifier
     */
    double train_binary_classifier(
        const torch::Tensor& X,
        const torch::Tensor& y_binary,
        const KernelParameters& params,
        DataConverter& converter,
        int class_idx);

    /**
     * @brief Get decision value for a single sample
     * @param sample Single feature tensor
     * @param model_idx Index of the model
     * @param converter Data converter
     * @return Decision value
     */
    double get_sample_decision_value(const torch::Tensor& sample, size_t model_idx, DataConverter& converter) const;

    /**
     * @brief Get probability estimate for a single sample
     * @param sample Single feature tensor
     * @param model_idx Index of the model
     * @param converter Data converter
     * @return Probability of positive class
     */
    double get_sample_probability(const torch::Tensor& sample, size_t model_idx, DataConverter& converter) const;

    /**
     * @brief Clean up all models
     */
    void cleanup_models();
};

/**
 * @brief One-vs-One (OvO) multiclass strategy
 */
class OneVsOneStrategy : public MulticlassStrategyBase {
public:
    /**
     * @brief Constructor
     */
    OneVsOneStrategy();

    /**
     * @brief Destructor
     */
    ~OneVsOneStrategy() override;

    TrainingMetrics fit(
        const torch::Tensor& X,
        const torch::Tensor& y,
        const KernelParameters& params,
        DataConverter& converter) override;

    std::vector<int> predict(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<std::vector<double>> predict_proba(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<std::vector<double>> decision_function(const torch::Tensor& X, DataConverter& converter) override;

    std::vector<int> get_classes() const override {
        return classes_;
    }

    bool supports_probability() const override;

    int get_n_classes() const override {
        return static_cast<int>(classes_.size());
    }

    MulticlassStrategy get_strategy_type() const override {
        return MulticlassStrategy::ONE_VS_ONE;
    }

private:
    std::vector<std::unique_ptr<svm_model>> svm_models_; ///< SVM models (one per pair)
    std::vector<std::unique_ptr<model>> linear_models_;  ///< Linear models (one per pair)
    std::vector<std::pair<int, int>> class_pairs_;       ///< Class pairs for each model
    std::vector<std::unique_ptr<DataConverter>>
        data_converters_;     ///< Data converters (one per pair) to keep training data alive
    KernelParameters params_; ///< Stored parameters
    SVMLibrary library_type_; ///< Which library is being used

    /**
     * @brief Extract samples for a specific class pair
     * @param X Feature tensor
     * @param y Label tensor
     * @param class1 First class
     * @param class2 Second class
     * @return Pair of (filtered_X, filtered_y)
     */
    std::pair<torch::Tensor, torch::Tensor>
    extract_binary_data(const torch::Tensor& X, const torch::Tensor& y, int class1, int class2);

    /**
     * @brief Train a single pairwise classifier
     * @param X Feature tensor
     * @param y Labels
     * @param class1 First class
     * @param class2 Second class
     * @param params Kernel parameters
     * @param converter Data converter
     * @param model_idx Index of the model being trained
     * @return Training time for this classifier
     */
    double train_pairwise_classifier(
        const torch::Tensor& X,
        const torch::Tensor& y,
        int class1,
        int class2,
        const KernelParameters& params,
        DataConverter& converter,
        int model_idx);

    /**
     * @brief Get decision value for a single sample
     * @param sample Single feature tensor
     * @param model_idx Index of the model
     * @return Decision value
     */
    double get_sample_decision_value(const torch::Tensor& sample, size_t model_idx) const;

    /**
     * @brief Voting mechanism for OvO predictions
     * @param decisions Matrix of pairwise decisions
     * @return Predicted class for each sample
     */
    std::vector<int> vote_predictions(const std::vector<std::vector<double>>& decisions);

    /**
     * @brief Clean up all models
     */
    void cleanup_models();
};

/**
 * @brief Factory function to create multiclass strategy
 * @param strategy Strategy type
 * @return Unique pointer to multiclass strategy
 */
std::unique_ptr<MulticlassStrategyBase> create_multiclass_strategy(MulticlassStrategy strategy);

} // namespace svm_classifier