#include "svm_classifier/data_converter.hpp"
#include "svm.h"        // libsvm
#include "linear.h"     // liblinear
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace svm_classifier {

    DataConverter::DataConverter()
        : n_features_(0)
        , n_samples_(0)
        , sparse_threshold_(1e-8)
    {
    }

    DataConverter::~DataConverter()
    {
        cleanup();
    }

    std::unique_ptr<svm_problem> DataConverter::to_svm_problem(const torch::Tensor& X,
        const torch::Tensor& y)
    {
        validate_tensors(X, y);

        auto X_cpu = ensure_cpu_tensor(X);

        n_samples_ = X_cpu.size(0);
        n_features_ = X_cpu.size(1);

        // Convert tensor data to svm_node structures
        svm_nodes_storage_ = tensor_to_svm_nodes(X_cpu);

        // Prepare pointers for svm_problem
        svm_x_space_.clear();
        svm_x_space_.reserve(n_samples_);

        for (auto& nodes : svm_nodes_storage_) {
            svm_x_space_.push_back(nodes.data());
        }

        // Extract labels if provided
        if (y.defined() && y.numel() > 0) {
            svm_y_space_ = extract_labels(y);
        } else {
            svm_y_space_.clear();
            svm_y_space_.resize(n_samples_, 0.0); // Dummy labels for prediction
        }

        // Create svm_problem
        auto problem = std::make_unique<svm_problem>();
        problem->l = n_samples_;
        problem->x = svm_x_space_.data();
        problem->y = svm_y_space_.data();

        return problem;
    }

    std::unique_ptr<problem> DataConverter::to_linear_problem(const torch::Tensor& X,
        const torch::Tensor& y)
    {
        validate_tensors(X, y);

        auto X_cpu = ensure_cpu_tensor(X);

        n_samples_ = X_cpu.size(0);
        n_features_ = X_cpu.size(1);

        // Convert tensor data to feature_node structures
        linear_nodes_storage_ = tensor_to_linear_nodes(X_cpu);

        // Prepare pointers for problem
        linear_x_space_.clear();
        linear_x_space_.reserve(n_samples_);

        for (auto& nodes : linear_nodes_storage_) {
            linear_x_space_.push_back(nodes.data());
        }

        // Extract labels if provided
        if (y.defined() && y.numel() > 0) {
            linear_y_space_ = extract_labels(y);
        } else {
            linear_y_space_.clear();
            linear_y_space_.resize(n_samples_, 0.0); // Dummy labels for prediction
        }

        // Create problem
        auto linear_problem = std::make_unique<problem>();
        linear_problem->l = n_samples_;
        linear_problem->n = n_features_;
        linear_problem->x = linear_x_space_.data();
        linear_problem->y = linear_y_space_.data();
        linear_problem->bias = 1.0; // Add bias term with value 1.0

        return linear_problem;
    }

    svm_node* DataConverter::to_svm_node(const torch::Tensor& sample)
    {
        validate_tensor_properties(sample, 1, "sample");

        auto sample_cpu = ensure_cpu_tensor(sample);
        single_svm_nodes_ = sample_to_svm_nodes(sample_cpu);

        return single_svm_nodes_.data();
    }

    feature_node* DataConverter::to_feature_node(const torch::Tensor& sample)
    {
        validate_tensor_properties(sample, 1, "sample");

        auto sample_cpu = ensure_cpu_tensor(sample);
        single_linear_nodes_ = sample_to_linear_nodes(sample_cpu);

        return single_linear_nodes_.data();
    }

    torch::Tensor DataConverter::from_predictions(const std::vector<double>& predictions)
    {
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        auto tensor = torch::zeros({ static_cast<int64_t>(predictions.size()) }, options);

        for (size_t i = 0; i < predictions.size(); ++i) {
            tensor[i] = static_cast<int>(predictions[i]);
        }

        return tensor;
    }

    torch::Tensor DataConverter::from_probabilities(const std::vector<std::vector<double>>& probabilities)
    {
        if (probabilities.empty()) {
            return torch::empty({ 0, 0 });
        }

        int n_samples = static_cast<int>(probabilities.size());
        int n_classes = static_cast<int>(probabilities[0].size());

        auto tensor = torch::zeros({ n_samples, n_classes }, torch::kFloat64);

        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_classes; ++j) {
                tensor[i][j] = probabilities[i][j];
            }
        }

        return tensor;
    }

    torch::Tensor DataConverter::from_decision_values(const std::vector<std::vector<double>>& decision_values)
    {
        if (decision_values.empty()) {
            return torch::empty({ 0, 0 });
        }

        int n_samples = static_cast<int>(decision_values.size());
        int n_values = static_cast<int>(decision_values[0].size());

        auto tensor = torch::zeros({ n_samples, n_values }, torch::kFloat64);

        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_values; ++j) {
                tensor[i][j] = decision_values[i][j];
            }
        }

        return tensor;
    }

    void DataConverter::validate_tensors(const torch::Tensor& X, const torch::Tensor& y)
    {
        validate_tensor_properties(X, 2, "X");

        if (y.defined() && y.numel() > 0) {
            validate_tensor_properties(y, 1, "y");

            // Check that number of samples match
            if (X.size(0) != y.size(0)) {
                throw std::invalid_argument(
                    "Number of samples in X (" + std::to_string(X.size(0)) +
                    ") does not match number of labels in y (" + std::to_string(y.size(0)) + ")"
                );
            }
        }

        // Check for reasonable dimensions
        if (X.size(0) == 0) {
            throw std::invalid_argument("X cannot have 0 samples");
        }

        if (X.size(1) == 0) {
            throw std::invalid_argument("X cannot have 0 features");
        }
    }

    void DataConverter::cleanup()
    {
        svm_nodes_storage_.clear();
        svm_x_space_.clear();
        svm_y_space_.clear();

        linear_nodes_storage_.clear();
        linear_x_space_.clear();
        linear_y_space_.clear();

        single_svm_nodes_.clear();
        single_linear_nodes_.clear();

        n_features_ = 0;
        n_samples_ = 0;
    }

    std::vector<std::vector<svm_node>> DataConverter::tensor_to_svm_nodes(const torch::Tensor& X)
    {
        std::vector<std::vector<svm_node>> nodes_storage;
        nodes_storage.reserve(X.size(0));

        auto X_acc = X.accessor<float, 2>();

        for (int i = 0; i < X.size(0); ++i) {
            nodes_storage.push_back(sample_to_svm_nodes(X[i]));
        }

        return nodes_storage;
    }

    std::vector<std::vector<feature_node>> DataConverter::tensor_to_linear_nodes(const torch::Tensor& X)
    {
        std::vector<std::vector<feature_node>> nodes_storage;
        nodes_storage.reserve(X.size(0));

        for (int i = 0; i < X.size(0); ++i) {
            nodes_storage.push_back(sample_to_linear_nodes(X[i]));
        }

        return nodes_storage;
    }

    std::vector<svm_node> DataConverter::sample_to_svm_nodes(const torch::Tensor& sample)
    {
        std::vector<svm_node> nodes;

        auto sample_acc = sample.accessor<float, 1>();

        // Reserve space (worst case: all features are non-sparse)
        nodes.reserve(sample.size(0) + 1); // +1 for terminator

        for (int j = 0; j < sample.size(0); ++j) {
            double value = static_cast<double>(sample_acc[j]);

            // Skip sparse features
            if (std::abs(value) > sparse_threshold_) {
                svm_node node;
                node.index = j + 1; // libsvm uses 1-based indexing
                node.value = value;
                nodes.push_back(node);
            }
        }

        // Add terminator
        svm_node terminator;
        terminator.index = -1;
        terminator.value = 0;
        nodes.push_back(terminator);

        return nodes;
    }

    std::vector<feature_node> DataConverter::sample_to_linear_nodes(const torch::Tensor& sample)
    {
        std::vector<feature_node> nodes;

        auto sample_acc = sample.accessor<float, 1>();

        // Reserve space (worst case: all features are non-sparse)
        nodes.reserve(sample.size(0) + 1); // +1 for terminator

        for (int j = 0; j < sample.size(0); ++j) {
            double value = static_cast<double>(sample_acc[j]);

            // Skip sparse features
            if (std::abs(value) > sparse_threshold_) {
                feature_node node;
                node.index = j + 1; // liblinear uses 1-based indexing
                node.value = value;
                nodes.push_back(node);
            }
        }

        // Add terminator
        feature_node terminator;
        terminator.index = -1;
        terminator.value = 0;
        nodes.push_back(terminator);

        return nodes;
    }

    std::vector<double> DataConverter::extract_labels(const torch::Tensor& y)
    {
        auto y_cpu = ensure_cpu_tensor(y);
        std::vector<double> labels;
        labels.reserve(y_cpu.size(0));

        // Handle different tensor types
        if (y_cpu.dtype() == torch::kInt32) {
            auto y_acc = y_cpu.accessor<int32_t, 1>();
            for (int i = 0; i < y_cpu.size(0); ++i) {
                labels.push_back(static_cast<double>(y_acc[i]));
            }
        } else if (y_cpu.dtype() == torch::kInt64) {
            auto y_acc = y_cpu.accessor<int64_t, 1>();
            for (int i = 0; i < y_cpu.size(0); ++i) {
                labels.push_back(static_cast<double>(y_acc[i]));
            }
        } else if (y_cpu.dtype() == torch::kFloat32) {
            auto y_acc = y_cpu.accessor<float, 1>();
            for (int i = 0; i < y_cpu.size(0); ++i) {
                labels.push_back(static_cast<double>(y_acc[i]));
            }
        } else if (y_cpu.dtype() == torch::kFloat64) {
            auto y_acc = y_cpu.accessor<double, 1>();
            for (int i = 0; i < y_cpu.size(0); ++i) {
                labels.push_back(y_acc[i]);
            }
        } else {
            throw std::invalid_argument("Unsupported label tensor dtype");
        }

        return labels;
    }

    torch::Tensor DataConverter::ensure_cpu_tensor(const torch::Tensor& tensor)
    {
        if (tensor.device().type() != torch::kCPU) {
            return tensor.to(torch::kCPU);
        }

        // Convert to float32 if not already
        if (tensor.dtype() != torch::kFloat32) {
            return tensor.to(torch::kFloat32);
        }

        return tensor;
    }

    void DataConverter::validate_tensor_properties(const torch::Tensor& tensor,
        int expected_dims,
        const std::string& name)
    {
        if (!tensor.defined()) {
            throw std::invalid_argument(name + " tensor is not defined");
        }

        if (tensor.dim() != expected_dims) {
            throw std::invalid_argument(
                name + " must have " + std::to_string(expected_dims) +
                " dimensions, got " + std::to_string(tensor.dim())
            );
        }

        if (tensor.numel() == 0) {
            throw std::invalid_argument(name + " tensor cannot be empty");
        }

        // Check for NaN or Inf values
        if (torch::any(torch::isnan(tensor)).item<bool>()) {
            throw std::invalid_argument(name + " contains NaN values");
        }

        if (torch::any(torch::isinf(tensor)).item<bool>()) {
            throw std::invalid_argument(name + " contains infinite values");
        }
    }

} // namespace svm_classifier