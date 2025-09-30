/**
 * @file test_kernel_parameters.cpp
 * @brief Unit tests for KernelParameters class
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <svm_classifier/kernel_parameters.hpp>

using namespace svm_classifier;
using json = nlohmann::json;

TEST_CASE("KernelParameters Default Constructor", "[unit][kernel_parameters]") {
    KernelParameters params;

    SECTION("Default values are set correctly") {
        REQUIRE(params.get_kernel_type() == KernelType::LINEAR);
        REQUIRE(params.get_C() == Catch::Approx(1.0));
        REQUIRE(params.get_tolerance() == Catch::Approx(1e-3));
        REQUIRE(params.get_probability() == false);
        REQUIRE(params.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_REST);
    }

    SECTION("Kernel-specific parameters have defaults") {
        REQUIRE(params.get_gamma() == Catch::Approx(-1.0)); // Auto gamma
        REQUIRE(params.get_degree() == 3);
        REQUIRE(params.get_coef0() == Catch::Approx(0.0));
        REQUIRE(params.get_cache_size() == Catch::Approx(200.0));
    }
}

TEST_CASE("KernelParameters JSON Constructor", "[unit][kernel_parameters]") {
    SECTION("Linear kernel configuration") {
        json config = {
            { "kernel", "linear" }, { "C", 10.0 }, { "tolerance", 1e-4 }, { "probability", true }
        };

        KernelParameters params(config);

        REQUIRE(params.get_kernel_type() == KernelType::LINEAR);
        REQUIRE(params.get_C() == Catch::Approx(10.0));
        REQUIRE(params.get_tolerance() == Catch::Approx(1e-4));
        REQUIRE(params.get_probability() == true);
    }

    SECTION("RBF kernel configuration") {
        json config = {
            { "kernel", "rbf" }, { "C", 1.0 }, { "gamma", 0.1 }, { "multiclass_strategy", "ovo" }
        };

        KernelParameters params(config);

        REQUIRE(params.get_kernel_type() == KernelType::RBF);
        REQUIRE(params.get_C() == Catch::Approx(1.0));
        REQUIRE(params.get_gamma() == Catch::Approx(0.1));
        REQUIRE(params.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_ONE);
    }

    SECTION("Polynomial kernel configuration") {
        json config = { { "kernel", "polynomial" },
                        { "C", 5.0 },
                        { "degree", 4 },
                        { "gamma", 0.5 },
                        { "coef0", 1.0 } };

        KernelParameters params(config);

        REQUIRE(params.get_kernel_type() == KernelType::POLYNOMIAL);
        REQUIRE(params.get_degree() == 4);
        REQUIRE(params.get_gamma() == Catch::Approx(0.5));
        REQUIRE(params.get_coef0() == Catch::Approx(1.0));
    }

    SECTION("Sigmoid kernel configuration") {
        json config = { { "kernel", "sigmoid" }, { "gamma", 0.01 }, { "coef0", -1.0 } };

        KernelParameters params(config);

        REQUIRE(params.get_kernel_type() == KernelType::SIGMOID);
        REQUIRE(params.get_gamma() == Catch::Approx(0.01));
        REQUIRE(params.get_coef0() == Catch::Approx(-1.0));
    }
}

TEST_CASE("KernelParameters Setters and Getters", "[unit][kernel_parameters]") {
    KernelParameters params;

    SECTION("Set and get C parameter") {
        params.set_C(5.0);
        REQUIRE(params.get_C() == Catch::Approx(5.0));

        // Test validation
        REQUIRE_THROWS_AS(params.set_C(-1.0), std::invalid_argument);
        REQUIRE_THROWS_AS(params.set_C(0.0), std::invalid_argument);
    }

    SECTION("Set and get gamma parameter") {
        params.set_gamma(0.25);
        REQUIRE(params.get_gamma() == Catch::Approx(0.25));

        // Negative values should be allowed (for auto gamma)
        params.set_gamma(-1.0);
        REQUIRE(params.get_gamma() == Catch::Approx(-1.0));
    }

    SECTION("Set and get degree parameter") {
        params.set_degree(5);
        REQUIRE(params.get_degree() == 5);

        // Test validation
        REQUIRE_THROWS_AS(params.set_degree(0), std::invalid_argument);
        REQUIRE_THROWS_AS(params.set_degree(-1), std::invalid_argument);
    }

    SECTION("Set and get tolerance") {
        params.set_tolerance(1e-6);
        REQUIRE(params.get_tolerance() == Catch::Approx(1e-6));

        // Test validation
        REQUIRE_THROWS_AS(params.set_tolerance(-1e-3), std::invalid_argument);
        REQUIRE_THROWS_AS(params.set_tolerance(0.0), std::invalid_argument);
    }

    SECTION("Set and get cache size") {
        params.set_cache_size(500.0);
        REQUIRE(params.get_cache_size() == Catch::Approx(500.0));

        // Test validation
        REQUIRE_THROWS_AS(params.set_cache_size(-100.0), std::invalid_argument);
    }
}

TEST_CASE("KernelParameters Validation", "[unit][kernel_parameters]") {
    SECTION("Valid linear kernel parameters") {
        KernelParameters params;
        params.set_kernel_type(KernelType::LINEAR);
        params.set_C(1.0);
        params.set_tolerance(1e-3);

        REQUIRE_NOTHROW(params.validate());
    }

    SECTION("Valid RBF kernel parameters") {
        KernelParameters params;
        params.set_kernel_type(KernelType::RBF);
        params.set_C(1.0);
        params.set_gamma(0.1);

        REQUIRE_NOTHROW(params.validate());
    }

    SECTION("Valid polynomial kernel parameters") {
        KernelParameters params;
        params.set_kernel_type(KernelType::POLYNOMIAL);
        params.set_C(1.0);
        params.set_degree(3);
        params.set_gamma(0.1);
        params.set_coef0(0.0);

        REQUIRE_NOTHROW(params.validate());
    }

    SECTION("Invalid parameters throw exceptions") {
        KernelParameters params;

        // Invalid C - should throw during parameter setting
        params.set_kernel_type(KernelType::LINEAR);
        REQUIRE_THROWS_AS(params.set_C(-1.0), std::invalid_argument);

        // Set valid C value
        params.set_C(1.0);

        // Invalid tolerance - should throw during parameter setting
        REQUIRE_THROWS_AS(params.set_tolerance(-1e-3), std::invalid_argument);
    }
}

TEST_CASE("KernelParameters JSON Serialization", "[unit][kernel_parameters]") {
    SECTION("Get parameters as JSON") {
        KernelParameters params;
        params.set_kernel_type(KernelType::RBF);
        params.set_C(2.0);
        params.set_gamma(0.5);
        params.set_probability(true);

        auto json_params = params.get_parameters();

        REQUIRE(json_params["kernel"] == "rbf");
        REQUIRE(json_params["C"] == Catch::Approx(2.0));
        REQUIRE(json_params["gamma"] == Catch::Approx(0.5));
        REQUIRE(json_params["probability"] == true);
    }

    SECTION("Round-trip JSON serialization") {
        json original_config = {
            { "kernel", "polynomial" }, { "C", 3.0 },         { "degree", 4 },
            { "gamma", 0.25 },          { "coef0", 1.5 },     { "multiclass_strategy", "ovo" },
            { "probability", true },    { "tolerance", 1e-5 }
        };

        KernelParameters params(original_config);
        auto serialized_config = params.get_parameters();

        // Create new parameters from serialized config
        KernelParameters params2(serialized_config);

        // Verify they match
        REQUIRE(params2.get_kernel_type() == params.get_kernel_type());
        REQUIRE(params2.get_C() == Catch::Approx(params.get_C()));
        REQUIRE(params2.get_degree() == params.get_degree());
        REQUIRE(params2.get_gamma() == Catch::Approx(params.get_gamma()));
        REQUIRE(params2.get_coef0() == Catch::Approx(params.get_coef0()));
        REQUIRE(params2.get_multiclass_strategy() == params.get_multiclass_strategy());
        REQUIRE(params2.get_probability() == params.get_probability());
        REQUIRE(params2.get_tolerance() == Catch::Approx(params.get_tolerance()));
    }
}

TEST_CASE("KernelParameters Default Parameters", "[unit][kernel_parameters]") {
    SECTION("Linear kernel defaults") {
        auto defaults = KernelParameters::get_default_parameters(KernelType::LINEAR);

        REQUIRE(defaults["kernel"] == "linear");
        REQUIRE(defaults["C"] == 1.0);
        REQUIRE(defaults["tolerance"] == 1e-3);
        REQUIRE(defaults["probability"] == false);
    }

    SECTION("RBF kernel defaults") {
        auto defaults = KernelParameters::get_default_parameters(KernelType::RBF);

        REQUIRE(defaults["kernel"] == "rbf");
        REQUIRE(defaults["gamma"] == -1.0); // Auto gamma
        REQUIRE(defaults["cache_size"] == 200.0);
    }

    SECTION("Polynomial kernel defaults") {
        auto defaults = KernelParameters::get_default_parameters(KernelType::POLYNOMIAL);

        REQUIRE(defaults["kernel"] == "polynomial");
        REQUIRE(defaults["degree"] == 3);
        REQUIRE(defaults["coef0"] == 0.0);
    }

    SECTION("Reset to defaults") {
        KernelParameters params;

        // Modify parameters
        params.set_kernel_type(KernelType::RBF);
        params.set_C(10.0);
        params.set_gamma(0.1);

        // Reset to defaults
        params.reset_to_defaults();

        // Should be back to RBF defaults
        REQUIRE(params.get_kernel_type() == KernelType::RBF);
        REQUIRE(params.get_C() == Catch::Approx(1.0));
        REQUIRE(params.get_gamma() == Catch::Approx(-1.0)); // Auto gamma
    }
}

TEST_CASE("KernelParameters Type Conversions", "[unit][kernel_parameters]") {
    SECTION("Kernel type to string conversion") {
        REQUIRE(kernel_type_to_string(KernelType::LINEAR) == "linear");
        REQUIRE(kernel_type_to_string(KernelType::RBF) == "rbf");
        REQUIRE(kernel_type_to_string(KernelType::POLYNOMIAL) == "polynomial");
        REQUIRE(kernel_type_to_string(KernelType::SIGMOID) == "sigmoid");
    }

    SECTION("String to kernel type conversion") {
        REQUIRE(string_to_kernel_type("linear") == KernelType::LINEAR);
        REQUIRE(string_to_kernel_type("rbf") == KernelType::RBF);
        REQUIRE(string_to_kernel_type("polynomial") == KernelType::POLYNOMIAL);
        REQUIRE(string_to_kernel_type("poly") == KernelType::POLYNOMIAL);
        REQUIRE(string_to_kernel_type("sigmoid") == KernelType::SIGMOID);

        REQUIRE_THROWS_AS(string_to_kernel_type("invalid"), std::invalid_argument);
    }

    SECTION("Multiclass strategy conversions") {
        REQUIRE(multiclass_strategy_to_string(MulticlassStrategy::ONE_VS_REST) == "ovr");
        REQUIRE(multiclass_strategy_to_string(MulticlassStrategy::ONE_VS_ONE) == "ovo");

        REQUIRE(string_to_multiclass_strategy("ovr") == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(string_to_multiclass_strategy("one_vs_rest") == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(string_to_multiclass_strategy("ovo") == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE(string_to_multiclass_strategy("one_vs_one") == MulticlassStrategy::ONE_VS_ONE);

        REQUIRE_THROWS_AS(string_to_multiclass_strategy("invalid"), std::invalid_argument);
    }

    SECTION("SVM library selection") {
        REQUIRE(get_svm_library(KernelType::LINEAR) == SVMLibrary::LIBLINEAR);
        REQUIRE(get_svm_library(KernelType::RBF) == SVMLibrary::LIBSVM);
        REQUIRE(get_svm_library(KernelType::POLYNOMIAL) == SVMLibrary::LIBSVM);
        REQUIRE(get_svm_library(KernelType::SIGMOID) == SVMLibrary::LIBSVM);
    }
}

TEST_CASE("KernelParameters Edge Cases", "[unit][kernel_parameters]") {
    SECTION("Empty JSON configuration") {
        json empty_config = json::object();

        // Should use all defaults
        REQUIRE_NOTHROW(KernelParameters(empty_config));

        KernelParameters params(empty_config);
        REQUIRE(params.get_kernel_type() == KernelType::LINEAR);
        REQUIRE(params.get_C() == Catch::Approx(1.0));
    }

    SECTION("Invalid JSON values") {
        json invalid_config = { { "kernel", "invalid_kernel" }, { "C", -1.0 } };

        REQUIRE_THROWS_AS(KernelParameters(invalid_config), std::invalid_argument);
    }

    SECTION("Partial JSON configuration") {
        json partial_config = {
            { "kernel", "rbf" }, { "C", 5.0 } // Missing gamma, should use default
        };

        KernelParameters params(partial_config);
        REQUIRE(params.get_kernel_type() == KernelType::RBF);
        REQUIRE(params.get_C() == Catch::Approx(5.0));
        REQUIRE(params.get_gamma() == Catch::Approx(-1.0)); // Default auto gamma
    }

    SECTION("Maximum and minimum valid values") {
        KernelParameters params;

        // Very small but valid C
        params.set_C(1e-10);
        REQUIRE(params.get_C() == Catch::Approx(1e-10));

        // Very large C
        params.set_C(1e10);
        REQUIRE(params.get_C() == Catch::Approx(1e10));

        // Very small tolerance
        params.set_tolerance(1e-15);
        REQUIRE(params.get_tolerance() == Catch::Approx(1e-15));
    }

    SECTION("set_gamma_auto method") {
        KernelParameters params;

        // Set gamma to a specific value first
        params.set_gamma(0.5);
        REQUIRE(params.get_gamma() == Catch::Approx(0.5));
        REQUIRE_FALSE(params.is_gamma_auto());

        // Set it to auto
        params.set_gamma_auto();
        REQUIRE(params.get_gamma() == Catch::Approx(-1.0));
        REQUIRE(params.is_gamma_auto());
    }
}