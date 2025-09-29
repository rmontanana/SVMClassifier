#pragma once

#include "types.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

// Forward declarations for libsvm and liblinear structures
struct svm_node;
struct svm_problem;
struct feature_node;
struct problem;

namespace svm_classifier {

    /**
     * @brief Data converter between libtorch tensors and SVM library formats
     *
     * This class handles the conversion between PyTorch tensors and the data structures
     * required by libsvm and liblinear libraries. It manages memory allocation and
     * provides efficient conversion methods.
     */
    class DataConverter {
    public:
        /**
         * @brief Default constructor
         */
        DataConverter();

        /**
         * @brief Destructor - cleans up allocated memory
         */
        ~DataConverter();

        /**
         * @brief Convert PyTorch tensors to libsvm format
         * @param X Feature tensor of shape (n_samples, n_features)
         * @param y Target tensor of shape (n_samples,) - optional for prediction
         * @return Pointer to svm_problem structure
         */
        std::unique_ptr<svm_problem> to_svm_problem(const torch::Tensor& X,
            const torch::Tensor& y = torch::Tensor());

        /**
         * @brief Convert PyTorch tensors to liblinear format
         * @param X Feature tensor of shape (n_samples, n_features)
         * @param y Target tensor of shape (n_samples,) - optional for prediction
         * @return Pointer to problem structure
         */
        std::unique_ptr<problem> to_linear_problem(const torch::Tensor& X,
            const torch::Tensor& y = torch::Tensor());

        /**
         * @brief Convert single sample to libsvm format
         * @param sample Feature tensor of shape (n_features,)
         * @return Pointer to svm_node array
         */
        std::vector<svm_node> to_svm_node(const torch::Tensor& sample);

        /**
         * @brief Convert single sample to liblinear format
         * @param sample Feature tensor of shape (n_features,)
         * @return Pointer to feature_node array
         */
        std::vector<feature_node> to_feature_node(const torch::Tensor& sample);

        /**
         * @brief Convert predictions back to PyTorch tensor
         * @param predictions Vector of predictions
         * @return PyTorch tensor with predictions
         */
        torch::Tensor from_predictions(const std::vector<double>& predictions);

        /**
         * @brief Convert probabilities back to PyTorch tensor
         * @param probabilities 2D vector of class probabilities
         * @return PyTorch tensor with probabilities of shape (n_samples, n_classes)
         */
        torch::Tensor from_probabilities(const std::vector<std::vector<double>>& probabilities);

        /**
         * @brief Convert decision values back to PyTorch tensor
         * @param decision_values 2D vector of decision function values
         * @return PyTorch tensor with decision values
         */
        torch::Tensor from_decision_values(const std::vector<std::vector<double>>& decision_values);

        /**
         * @brief Validate input tensors
         * @param X Feature tensor
         * @param y Target tensor (optional)
         * @throws std::invalid_argument if tensors are invalid
         */
        void validate_tensors(const torch::Tensor& X, const torch::Tensor& y = torch::Tensor());

        /**
         * @brief Get number of features from last conversion
         * @return Number of features
         */
        int get_n_features() const { return n_features_; }

        /**
         * @brief Get number of samples from last conversion
         * @return Number of samples
         */
        int get_n_samples() const { return n_samples_; }

        /**
         * @brief Clean up all allocated memory
         */
        void cleanup();

        /**
         * @brief Set sparse threshold (features with absolute value below this are ignored)
         * @param threshold Sparse threshold (default: 1e-8)
         */
        void set_sparse_threshold(double threshold) { sparse_threshold_ = threshold; }

        /**
         * @brief Get sparse threshold
         * @return Current sparse threshold
         */
        double get_sparse_threshold() const { return sparse_threshold_; }

    private:
        int n_features_;     ///< Number of features
        int n_samples_;      ///< Number of samples
        double sparse_threshold_;  ///< Threshold for sparse features

        // Memory management for libsvm structures
        std::vector<std::vector<svm_node>> svm_nodes_storage_;
        std::vector<svm_node*> svm_x_space_;
        std::vector<double> svm_y_space_;

        // Memory management for liblinear structures
        std::vector<std::vector<feature_node>> linear_nodes_storage_;
        std::vector<feature_node*> linear_x_space_;
        std::vector<double> linear_y_space_;



        /**
         * @brief Convert tensor data to libsvm nodes for multiple samples
         * @param X Feature tensor
         * @return Vector of svm_node vectors
         */
        std::vector<std::vector<svm_node>> tensor_to_svm_nodes(const torch::Tensor& X);

        /**
         * @brief Convert tensor data to liblinear nodes for multiple samples
         * @param X Feature tensor
         * @return Vector of feature_node vectors
         */
        std::vector<std::vector<feature_node>> tensor_to_linear_nodes(const torch::Tensor& X);

        /**
         * @brief Convert single tensor sample to svm_node vector
         * @param sample Feature tensor of shape (n_features,)
         * @return Vector of svm_node structures
         */
        std::vector<svm_node> sample_to_svm_nodes(const torch::Tensor& sample);

        /**
         * @brief Convert single tensor sample to feature_node vector
         * @param sample Feature tensor of shape (n_features,)
         * @return Vector of feature_node structures
         */
        std::vector<feature_node> sample_to_linear_nodes(const torch::Tensor& sample);

        /**
         * @brief Extract labels from target tensor
         * @param y Target tensor
         * @return Vector of double labels
         */
        std::vector<double> extract_labels(const torch::Tensor& y);

        /**
         * @brief Check if tensor is on CPU and convert if necessary
         * @param tensor Input tensor
         * @return Tensor guaranteed to be on CPU
         */
        torch::Tensor ensure_cpu_tensor(const torch::Tensor& tensor);

        /**
         * @brief Validate tensor dimensions and data type
         * @param tensor Tensor to validate
         * @param expected_dims Expected number of dimensions
         * @param name Tensor name for error messages
         */
        void validate_tensor_properties(const torch::Tensor& tensor, int expected_dims, const std::string& name);
    };

} // namespace svm_classifier