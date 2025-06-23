/**
 * @file test_data_converter.cpp
 * @brief Unit tests for DataConverter class
 */

#include <catch2/catch_all.hpp>
#include <svm_classifier/data_converter.hpp>
#include <torch/torch.h>

// Include the actual headers for complete struct definitions
#include "svm.h"        // libsvm structures
#include "linear.h"     // liblinear structures

using namespace svm_classifier;

TEST_CASE("DataConverter Basic Functionality", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Tensor validation")
    {
        // Valid 2D tensor
        auto X = torch::randn({ 10, 5 });
        auto y = torch::randint(0, 3, { 10 });

        REQUIRE_NOTHROW(converter.validate_tensors(X, y));

        // Invalid dimensions
        auto X_invalid = torch::randn({ 10 });  // 1D instead of 2D
        REQUIRE_THROWS_AS(converter.validate_tensors(X_invalid, y), std::invalid_argument);

        // Mismatched samples
        auto y_invalid = torch::randint(0, 3, { 5 });  // Different number of samples
        REQUIRE_THROWS_AS(converter.validate_tensors(X, y_invalid), std::invalid_argument);

        // Empty tensors
        auto X_empty = torch::empty({ 0, 5 });
        REQUIRE_THROWS_AS(converter.validate_tensors(X_empty, y), std::invalid_argument);

        auto X_no_features = torch::empty({ 10, 0 });
        REQUIRE_THROWS_AS(converter.validate_tensors(X_no_features, y), std::invalid_argument);
    }

    SECTION("NaN and Inf detection")
    {
        auto X = torch::randn({ 5, 3 });
        auto y = torch::randint(0, 2, { 5 });

        // Introduce NaN
        X[0][0] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS_AS(converter.validate_tensors(X, y), std::invalid_argument);

        // Introduce Inf
        X[0][0] = std::numeric_limits<float>::infinity();
        REQUIRE_THROWS_AS(converter.validate_tensors(X, y), std::invalid_argument);
    }
}

TEST_CASE("DataConverter SVM Problem Conversion", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Basic conversion")
    {
        auto X = torch::tensor({ {1.0, 2.0, 3.0},
                               {4.0, 5.0, 6.0},
                               {7.0, 8.0, 9.0} });
        auto y = torch::tensor({ 0, 1, 2 });

        auto problem = converter.to_svm_problem(X, y);

        REQUIRE(problem != nullptr);
        REQUIRE(problem->l == 3);  // Number of samples
        REQUIRE(converter.get_n_samples() == 3);
        REQUIRE(converter.get_n_features() == 3);

        // Check labels
        REQUIRE(problem->y[0] == Catch::Approx(0.0));
        REQUIRE(problem->y[1] == Catch::Approx(1.0));
        REQUIRE(problem->y[2] == Catch::Approx(2.0));
    }

    SECTION("Conversion without labels")
    {
        auto X = torch::tensor({ {1.0, 2.0},
                               {3.0, 4.0} });

        auto problem = converter.to_svm_problem(X);

        REQUIRE(problem != nullptr);
        REQUIRE(problem->l == 2);
        REQUIRE(converter.get_n_samples() == 2);
        REQUIRE(converter.get_n_features() == 2);
    }

    SECTION("Sparse features handling")
    {
        // Create tensor with some very small values (should be treated as sparse)
        auto X = torch::tensor({ {1.0, 1e-10, 2.0},
                               {0.0, 3.0, 1e-9} });
        auto y = torch::tensor({ 0, 1 });

        converter.set_sparse_threshold(1e-8);
        auto problem = converter.to_svm_problem(X, y);

        REQUIRE(problem != nullptr);
        REQUIRE(problem->l == 2);

        // The very small values should be ignored in the sparse representation
        // This is implementation-specific and would need to check the actual svm_node structure
    }
}

TEST_CASE("DataConverter Linear Problem Conversion", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Basic conversion")
    {
        auto X = torch::tensor({ {1.0, 2.0},
                               {3.0, 4.0},
                               {5.0, 6.0} });
        auto y = torch::tensor({ -1, 1, -1 });

        auto problem = converter.to_linear_problem(X, y);

        REQUIRE(problem != nullptr);
        REQUIRE(problem->l == 3);      // Number of samples
        REQUIRE(problem->n == 2);      // Number of features
        REQUIRE(problem->bias == -1);  // No bias term

        // Check labels
        REQUIRE(problem->y[0] == Catch::Approx(-1.0));
        REQUIRE(problem->y[1] == Catch::Approx(1.0));
        REQUIRE(problem->y[2] == Catch::Approx(-1.0));
    }

    SECTION("Different tensor dtypes")
    {
        // Test with different data types
        auto X_int = torch::tensor({ {1, 2}, {3, 4} }, torch::kInt32);
        auto y_int = torch::tensor({ 0, 1 }, torch::kInt32);

        REQUIRE_NOTHROW(converter.to_linear_problem(X_int, y_int));

        auto X_double = torch::tensor({ {1.0, 2.0}, {3.0, 4.0} }, torch::kFloat64);
        auto y_double = torch::tensor({ 0.0, 1.0 }, torch::kFloat64);

        REQUIRE_NOTHROW(converter.to_linear_problem(X_double, y_double));
    }
}

TEST_CASE("DataConverter Single Sample Conversion", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("SVM node conversion")
    {
        auto sample = torch::tensor({ 1.0, 0.0, 3.0, 0.0, 5.0 });

        auto nodes = converter.to_svm_node(sample);

        REQUIRE(nodes != nullptr);

        // Should have non-zero features plus terminator
        // This is implementation-specific and depends on sparse handling
    }

    SECTION("Feature node conversion")
    {
        auto sample = torch::tensor({ 2.0, 4.0, 6.0 });

        auto nodes = converter.to_feature_node(sample);

        REQUIRE(nodes != nullptr);
    }

    SECTION("Invalid single sample")
    {
        auto invalid_sample = torch::tensor({ {1.0, 2.0} }); // 2D instead of 1D

        REQUIRE_THROWS_AS(converter.to_svm_node(invalid_sample), std::invalid_argument);
        REQUIRE_THROWS_AS(converter.to_feature_node(invalid_sample), std::invalid_argument);
    }
}

TEST_CASE("DataConverter Result Conversion", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Predictions conversion")
    {
        std::vector<double> predictions = { 0.0, 1.0, 2.0, 1.0, 0.0 };

        auto tensor = converter.from_predictions(predictions);

        REQUIRE(tensor.dtype() == torch::kInt32);
        REQUIRE(tensor.size(0) == 5);

        for (int i = 0; i < 5; ++i) {
            REQUIRE(tensor[i].item<int>() == static_cast<int>(predictions[i]));
        }
    }

    SECTION("Probabilities conversion")
    {
        std::vector<std::vector<double>> probabilities = {
            {0.7, 0.2, 0.1},
            {0.1, 0.8, 0.1},
            {0.3, 0.3, 0.4}
        };

        auto tensor = converter.from_probabilities(probabilities);

        REQUIRE(tensor.dtype() == torch::kFloat64);
        REQUIRE(tensor.size(0) == 3);  // 3 samples
        REQUIRE(tensor.size(1) == 3);  // 3 classes

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(tensor[i][j].item<double>() == Catch::Approx(probabilities[i][j]));
            }
        }
    }

    SECTION("Decision values conversion")
    {
        std::vector<std::vector<double>> decision_values = {
            {1.5, -0.5},
            {-1.0, 2.0}
        };

        auto tensor = converter.from_decision_values(decision_values);

        REQUIRE(tensor.dtype() == torch::kFloat64);
        REQUIRE(tensor.size(0) == 2);  // 2 samples
        REQUIRE(tensor.size(1) == 2);  // 2 decision values

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                REQUIRE(tensor[i][j].item<double>() == Catch::Approx(decision_values[i][j]));
            }
        }
    }

    SECTION("Empty results")
    {
        std::vector<double> empty_predictions;
        auto tensor = converter.from_predictions(empty_predictions);
        REQUIRE(tensor.size(0) == 0);

        std::vector<std::vector<double>> empty_probabilities;
        auto prob_tensor = converter.from_probabilities(empty_probabilities);
        REQUIRE(prob_tensor.size(0) == 0);
        REQUIRE(prob_tensor.size(1) == 0);
    }
}

TEST_CASE("DataConverter Memory Management", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Cleanup functionality")
    {
        auto X = torch::randn({ 100, 50 });
        auto y = torch::randint(0, 5, { 100 });

        // Convert to problems
        auto svm_problem = converter.to_svm_problem(X, y);
        auto linear_problem = converter.to_linear_problem(X, y);

        REQUIRE(converter.get_n_samples() == 100);
        REQUIRE(converter.get_n_features() == 50);

        // Cleanup
        converter.cleanup();

        REQUIRE(converter.get_n_samples() == 0);
        REQUIRE(converter.get_n_features() == 0);
    }

    SECTION("Multiple conversions")
    {
        // Test that converter can handle multiple conversions
        for (int i = 0; i < 5; ++i) {
            auto X = torch::randn({ 10, 3 });
            auto y = torch::randint(0, 2, { 10 });

            REQUIRE_NOTHROW(converter.to_svm_problem(X, y));
            REQUIRE_NOTHROW(converter.to_linear_problem(X, y));
        }
    }
}

TEST_CASE("DataConverter Sparse Threshold", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("Sparse threshold configuration")
    {
        REQUIRE(converter.get_sparse_threshold() == Catch::Approx(1e-8));

        converter.set_sparse_threshold(1e-6);
        REQUIRE(converter.get_sparse_threshold() == Catch::Approx(1e-6));

        converter.set_sparse_threshold(0.0);
        REQUIRE(converter.get_sparse_threshold() == Catch::Approx(0.0));
    }

    SECTION("Sparse threshold effect")
    {
        auto X = torch::tensor({ {1.0, 1e-7, 1e-5},
                               {1e-9, 2.0, 1e-4} });
        auto y = torch::tensor({ 0, 1 });

        // With default threshold (1e-8), 1e-9 should be ignored
        converter.set_sparse_threshold(1e-8);
        auto problem1 = converter.to_svm_problem(X, y);

        // With larger threshold (1e-6), both 1e-7 and 1e-9 should be ignored
        converter.set_sparse_threshold(1e-6);
        auto problem2 = converter.to_svm_problem(X, y);

        // Both should succeed but might have different sparse representations
        REQUIRE(problem1 != nullptr);
        REQUIRE(problem2 != nullptr);
    }
}

TEST_CASE("DataConverter Device Handling", "[unit][data_converter]")
{
    DataConverter converter;

    SECTION("CPU tensors")
    {
        auto X = torch::randn({ 5, 3 }, torch::device(torch::kCPU));
        auto y = torch::randint(0, 2, { 5 }, torch::device(torch::kCPU));

        REQUIRE_NOTHROW(converter.to_svm_problem(X, y));
    }

    SECTION("GPU tensors (if available)")
    {
        if (torch::cuda::is_available()) {
            auto X = torch::randn({ 5, 3 }, torch::device(torch::kCUDA));
            auto y = torch::randint(0, 2, { 5 }, torch::device(torch::kCUDA));

            // Should work by automatically moving to CPU
            REQUIRE_NOTHROW(converter.to_svm_problem(X, y));
        }
    }

    SECTION("Mixed device tensors")
    {
        auto X = torch::randn({ 5, 3 }, torch::device(torch::kCPU));

        if (torch::cuda::is_available()) {
            auto y = torch::randint(0, 2, { 5 }, torch::device(torch::kCUDA));

            // Should work by moving both to CPU
            REQUIRE_NOTHROW(converter.to_svm_problem(X, y));
        }
    }
}