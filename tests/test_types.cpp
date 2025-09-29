/**
 * @file test_types.cpp
 * @brief Unit tests for types and utility functions
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <svm_classifier/types.hpp>

using namespace svm_classifier;

TEST_CASE("Kernel type string conversion", "[unit][types]")
{
    SECTION("kernel_type_to_string converts all valid kernel types")
    {
        REQUIRE(kernel_type_to_string(KernelType::LINEAR) == "linear");
        REQUIRE(kernel_type_to_string(KernelType::RBF) == "rbf");
        REQUIRE(kernel_type_to_string(KernelType::POLYNOMIAL) == "polynomial");
        REQUIRE(kernel_type_to_string(KernelType::SIGMOID) == "sigmoid");
    }

    SECTION("kernel_type_to_string handles invalid enum value")
    {
        // Cast an invalid value to KernelType to test the default case
        KernelType invalid_kernel = static_cast<KernelType>(999);
        REQUIRE(kernel_type_to_string(invalid_kernel) == "unknown");
    }

    SECTION("string_to_kernel_type converts all valid strings")
    {
        REQUIRE(string_to_kernel_type("linear") == KernelType::LINEAR);
        REQUIRE(string_to_kernel_type("rbf") == KernelType::RBF);
        REQUIRE(string_to_kernel_type("polynomial") == KernelType::POLYNOMIAL);
        REQUIRE(string_to_kernel_type("poly") == KernelType::POLYNOMIAL);
        REQUIRE(string_to_kernel_type("sigmoid") == KernelType::SIGMOID);
    }

    SECTION("string_to_kernel_type throws on invalid string")
    {
        REQUIRE_THROWS_AS(string_to_kernel_type("invalid"), std::invalid_argument);
        REQUIRE_THROWS_AS(string_to_kernel_type("unknown"), std::invalid_argument);
    }
}

TEST_CASE("Multiclass strategy string conversion", "[unit][types]")
{
    SECTION("multiclass_strategy_to_string converts all valid strategies")
    {
        REQUIRE(multiclass_strategy_to_string(MulticlassStrategy::ONE_VS_REST) == "ovr");
        REQUIRE(multiclass_strategy_to_string(MulticlassStrategy::ONE_VS_ONE) == "ovo");
    }

    SECTION("multiclass_strategy_to_string handles invalid enum value")
    {
        // Cast an invalid value to MulticlassStrategy to test the default case
        MulticlassStrategy invalid_strategy = static_cast<MulticlassStrategy>(999);
        REQUIRE(multiclass_strategy_to_string(invalid_strategy) == "unknown");
    }

    SECTION("string_to_multiclass_strategy converts all valid strings")
    {
        REQUIRE(string_to_multiclass_strategy("ovr") == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(string_to_multiclass_strategy("one_vs_rest") == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(string_to_multiclass_strategy("ovo") == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE(string_to_multiclass_strategy("one_vs_one") == MulticlassStrategy::ONE_VS_ONE);
    }

    SECTION("string_to_multiclass_strategy throws on invalid string")
    {
        REQUIRE_THROWS_AS(string_to_multiclass_strategy("invalid"), std::invalid_argument);
        REQUIRE_THROWS_AS(string_to_multiclass_strategy("unknown"), std::invalid_argument);
    }
}

TEST_CASE("SVM library selection", "[unit][types]")
{
    SECTION("get_svm_library selects correct library for kernel types")
    {
        REQUIRE(get_svm_library(KernelType::LINEAR) == SVMLibrary::LIBLINEAR);
        REQUIRE(get_svm_library(KernelType::RBF) == SVMLibrary::LIBSVM);
        REQUIRE(get_svm_library(KernelType::POLYNOMIAL) == SVMLibrary::LIBSVM);
        REQUIRE(get_svm_library(KernelType::SIGMOID) == SVMLibrary::LIBSVM);
    }
}