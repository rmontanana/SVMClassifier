#include "svm_classifier/kernel_parameters.hpp"

#include <cmath>
#include <stdexcept>

namespace svm_classifier {

namespace {
// Helper function to validate and extract numeric value from JSON
template <typename T>
T get_numeric_value(const nlohmann::json& config, const std::string& key, const std::string& type_name) {
    if (!config[key].is_number()) {
        throw std::invalid_argument(key + " must be a " + type_name);
    }
    return config[key].get<T>();
}

// Helper function to validate and extract integer value from JSON
int get_integer_value(const nlohmann::json& config, const std::string& key) {
    if (!config[key].is_number_integer()) {
        throw std::invalid_argument(key + " must be an integer");
    }
    return config[key].get<int>();
}

// Helper function to validate and extract boolean value from JSON
bool get_boolean_value(const nlohmann::json& config, const std::string& key) {
    if (!config[key].is_boolean()) {
        throw std::invalid_argument(key + " must be a boolean");
    }
    return config[key].get<bool>();
}

// Helper function to validate and extract string value from JSON
std::string get_string_value(const nlohmann::json& config, const std::string& key) {
    if (!config[key].is_string()) {
        throw std::invalid_argument(key + " must be a string");
    }
    return config[key].get<std::string>();
}

// Helper function to validate gamma (used in multiple places)
bool is_valid_gamma(double gamma) {
    return gamma > 0.0 || gamma == -1.0;
}
} // anonymous namespace

KernelParameters::KernelParameters()
    : kernel_type_(KernelType::LINEAR), multiclass_strategy_(MulticlassStrategy::ONE_VS_REST), C_(1.0),
      tolerance_(1e-3), max_iterations_(-1), probability_(false), cache_size_(200.0), gamma_(-1.0) // Auto gamma
      ,
      degree_(3), coef0_(0.0) {}

KernelParameters::KernelParameters(const nlohmann::json& config) : KernelParameters() {
    set_parameters(config);
}

void KernelParameters::set_parameters(const nlohmann::json& config) {
    // Set kernel type first as it affects validation
    if (config.contains("kernel")) {
        set_kernel_type(string_to_kernel_type(get_string_value(config, "kernel")));
    }

    // Set multiclass strategy
    if (config.contains("multiclass_strategy")) {
        set_multiclass_strategy(string_to_multiclass_strategy(get_string_value(config, "multiclass_strategy")));
    }

    // Set common parameters
    if (config.contains("C")) {
        set_C(get_numeric_value<double>(config, "C", "number"));
    }

    if (config.contains("tolerance")) {
        set_tolerance(get_numeric_value<double>(config, "tolerance", "number"));
    }

    if (config.contains("max_iterations")) {
        set_max_iterations(get_integer_value(config, "max_iterations"));
    }

    if (config.contains("probability")) {
        set_probability(get_boolean_value(config, "probability"));
    }

    // Set kernel-specific parameters
    if (config.contains("gamma")) {
        if (config["gamma"].is_string() && config["gamma"] == "auto") {
            set_gamma(-1.0); // Auto gamma
        } else {
            set_gamma(get_numeric_value<double>(config, "gamma", "number"));
        }
    }

    if (config.contains("degree")) {
        set_degree(get_integer_value(config, "degree"));
    }

    if (config.contains("coef0")) {
        set_coef0(get_numeric_value<double>(config, "coef0", "number"));
    }

    if (config.contains("cache_size")) {
        set_cache_size(get_numeric_value<double>(config, "cache_size", "number"));
    }

    // Validate all parameters
    validate();
}

nlohmann::json KernelParameters::get_parameters() const {
    nlohmann::json params = { { "kernel", kernel_type_to_string(kernel_type_) },
                              { "multiclass_strategy", multiclass_strategy_to_string(multiclass_strategy_) },
                              { "C", C_ },
                              { "tolerance", tolerance_ },
                              { "max_iterations", max_iterations_ },
                              { "probability", probability_ },
                              { "cache_size", cache_size_ } };

    // Add kernel-specific parameters
    switch (kernel_type_) {
    case KernelType::LINEAR:
        // No additional parameters for linear kernel
        break;

    case KernelType::RBF:
        if (is_gamma_auto()) {
            params["gamma"] = "auto";
        } else {
            params["gamma"] = gamma_;
        }
        break;

    case KernelType::POLYNOMIAL:
        params["degree"] = degree_;
        if (is_gamma_auto()) {
            params["gamma"] = "auto";
        } else {
            params["gamma"] = gamma_;
        }
        params["coef0"] = coef0_;
        break;

    case KernelType::SIGMOID:
        if (is_gamma_auto()) {
            params["gamma"] = "auto";
        } else {
            params["gamma"] = gamma_;
        }
        params["coef0"] = coef0_;
        break;
    }

    return params;
}

void KernelParameters::set_kernel_type(KernelType kernel) {
    kernel_type_ = kernel;

    // Reset kernel-specific parameters to defaults when kernel changes
    auto defaults = get_default_parameters(kernel);

    if (defaults.contains("gamma")) {
        gamma_ = defaults["gamma"];
    }
    if (defaults.contains("degree")) {
        degree_ = defaults["degree"];
    }
    if (defaults.contains("coef0")) {
        coef0_ = defaults["coef0"];
    }
}

void KernelParameters::set_C(double c) {
    if (c <= 0.0) {
        throw std::invalid_argument("C must be positive (C > 0)");
    }
    C_ = c;
}

void KernelParameters::set_gamma(double gamma) {
    if (!is_valid_gamma(gamma)) {
        throw std::invalid_argument("Gamma must be positive or -1 for auto");
    }
    gamma_ = gamma;
}

void KernelParameters::set_degree(int degree) {
    if (degree < 1) {
        throw std::invalid_argument("Degree must be >= 1");
    }
    degree_ = degree;
}

void KernelParameters::set_coef0(double coef0) {
    coef0_ = coef0;
}

void KernelParameters::set_tolerance(double tol) {
    if (tol <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive (tolerance > 0)");
    }
    tolerance_ = tol;
}

void KernelParameters::set_max_iterations(int max_iter) {
    if (max_iter <= 0 && max_iter != -1) {
        throw std::invalid_argument("Max iterations must be positive or -1 for no limit");
    }
    max_iterations_ = max_iter;
}

void KernelParameters::set_cache_size(double cache_size) {
    if (cache_size < 0.0) {
        throw std::invalid_argument("Cache size must be non-negative");
    }
    cache_size_ = cache_size;
}

void KernelParameters::set_probability(bool probability) {
    probability_ = probability;
}

void KernelParameters::set_multiclass_strategy(MulticlassStrategy strategy) {
    multiclass_strategy_ = strategy;
}

void KernelParameters::validate() const {
    // Validate common parameters
    if (C_ <= 0.0) {
        throw std::invalid_argument("C must be positive");
    }

    if (tolerance_ <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }

    if (max_iterations_ <= 0 && max_iterations_ != -1) {
        throw std::invalid_argument("Max iterations must be positive or -1");
    }

    if (cache_size_ < 0.0) {
        throw std::invalid_argument("Cache size must be non-negative");
    }

    // Validate kernel-specific parameters
    validate_kernel_parameters();
}

void KernelParameters::validate_kernel_parameters() const {
    switch (kernel_type_) {
    case KernelType::LINEAR:
        // Linear kernel has no additional parameters to validate
        break;

    case KernelType::RBF:
        if (!is_valid_gamma(gamma_)) {
            throw std::invalid_argument("RBF kernel gamma must be positive or auto (-1)");
        }
        break;

    case KernelType::POLYNOMIAL:
        if (degree_ < 1) {
            throw std::invalid_argument("Polynomial degree must be >= 1");
        }
        if (!is_valid_gamma(gamma_)) {
            throw std::invalid_argument("Polynomial kernel gamma must be positive or auto (-1)");
        }
        // coef0 can be any real number
        break;

    case KernelType::SIGMOID:
        if (!is_valid_gamma(gamma_)) {
            throw std::invalid_argument("Sigmoid kernel gamma must be positive or auto (-1)");
        }
        // coef0 can be any real number
        break;
    }
}

nlohmann::json KernelParameters::get_default_parameters(KernelType kernel) {
    nlohmann::json defaults = { { "C", 1.0 },
                                { "tolerance", 1e-3 },
                                { "max_iterations", -1 },
                                { "probability", false },
                                { "multiclass_strategy", "ovr" },
                                { "cache_size", 200.0 } };

    switch (kernel) {
    case KernelType::LINEAR:
        defaults["kernel"] = "linear";
        break;

    case KernelType::RBF:
        defaults["kernel"] = "rbf";
        defaults["gamma"] = -1.0; // Auto gamma
        break;

    case KernelType::POLYNOMIAL:
        defaults["kernel"] = "polynomial";
        defaults["degree"] = 3;
        defaults["gamma"] = -1.0; // Auto gamma
        defaults["coef0"] = 0.0;
        break;

    case KernelType::SIGMOID:
        defaults["kernel"] = "sigmoid";
        defaults["gamma"] = -1.0; // Auto gamma
        defaults["coef0"] = 0.0;
        break;
    }

    return defaults;
}

void KernelParameters::reset_to_defaults() {
    auto defaults = get_default_parameters(kernel_type_);
    set_parameters(defaults);
}

void KernelParameters::set_gamma_auto() {
    gamma_ = -1.0;
}

} // namespace svm_classifier