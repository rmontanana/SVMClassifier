#pragma once

#include "types.hpp"

#include <nlohmann/json.hpp>

namespace svm_classifier {

/**
 * @brief Kernel parameters configuration class
 *
 * This class manages all parameters for SVM kernels including kernel type,
 * regularization parameters, optimization settings, and kernel-specific parameters.
 */
class KernelParameters {
public:
    /**
     * @brief Default constructor with default parameters
     */
    KernelParameters();

    /**
     * @brief Constructor with JSON configuration
     * @param config JSON configuration object
     */
    explicit KernelParameters(const nlohmann::json& config);

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

    // Kernel type
    void set_kernel_type(KernelType kernel);
    KernelType get_kernel_type() const {
        return kernel_type_;
    }

    // Multiclass strategy
    void set_multiclass_strategy(MulticlassStrategy strategy);
    MulticlassStrategy get_multiclass_strategy() const {
        return multiclass_strategy_;
    }

    // Common parameters
    void set_C(double c);
    double get_C() const {
        return C_;
    }

    void set_tolerance(double tol);
    double get_tolerance() const {
        return tolerance_;
    }

    void set_max_iterations(int max_iter);
    int get_max_iterations() const {
        return max_iterations_;
    }

    void set_probability(bool probability);
    bool get_probability() const {
        return probability_;
    }

    void set_cache_size(double cache_size);
    double get_cache_size() const {
        return cache_size_;
    }

    // Kernel-specific parameters
    void set_gamma(double gamma);
    double get_gamma() const {
        return gamma_;
    }
    bool is_gamma_auto() const {
        return gamma_ == -1.0;
    }
    void set_gamma_auto();

    void set_degree(int degree);
    int get_degree() const {
        return degree_;
    }

    void set_coef0(double coef0);
    double get_coef0() const {
        return coef0_;
    }

    /**
     * @brief Validate all parameters for consistency
     * @throws std::invalid_argument if parameters are invalid
     */
    void validate() const;

    /**
     * @brief Get default parameters for a specific kernel type
     * @param kernel Kernel type
     * @return JSON object with default parameters
     */
    static nlohmann::json get_default_parameters(KernelType kernel);

    /**
     * @brief Reset all parameters to defaults for current kernel type
     */
    void reset_to_defaults();

private:
    KernelType kernel_type_;                 ///< Kernel type
    MulticlassStrategy multiclass_strategy_; ///< Multiclass strategy

    // Common parameters
    double C_;           ///< Regularization parameter
    double tolerance_;   ///< Convergence tolerance
    int max_iterations_; ///< Maximum iterations (-1 for no limit)
    bool probability_;   ///< Enable probability estimates
    double cache_size_;  ///< Cache size in MB

    // Kernel-specific parameters
    double gamma_; ///< Gamma parameter (-1 for auto)
    int degree_;   ///< Polynomial degree
    double coef0_; ///< Independent term in polynomial/sigmoid

    /**
     * @brief Validate kernel-specific parameters
     * @throws std::invalid_argument if kernel parameters are invalid
     */
    void validate_kernel_parameters() const;
};

} // namespace svm_classifier