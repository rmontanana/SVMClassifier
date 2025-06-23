/**
 * @file test_multiclass_strategy.cpp
 * @brief Unit tests for multiclass strategy classes
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <svm_classifier/multiclass_strategy.hpp>
#include <svm_classifier/kernel_parameters.hpp>
#include <svm_classifier/data_converter.hpp>
#include <torch/torch.h>

using namespace svm_classifier;

/**
 * @brief Generate simple test data for multiclass testing
 */
std::pair<torch::Tensor, torch::Tensor> generate_multiclass_data(int n_samples = 60,
    int n_features = 2,
    int n_classes = 3,
    int seed = 42)
{
    torch::manual_seed(seed);

    auto X = torch::randn({ n_samples, n_features });
    auto y = torch::randint(0, n_classes, { n_samples });

    // Create some structure in the data
    for (int i = 0; i < n_samples; ++i) {
        int class_label = y[i].item<int>();
        // Add class-specific bias to make classification easier
        X[i] += class_label * 0.5;
    }

    return { X, y };
}

TEST_CASE("MulticlassStrategy Factory Function", "[unit][multiclass_strategy]")
{
    SECTION("Create One-vs-Rest strategy")
    {
        auto strategy = create_multiclass_strategy(MulticlassStrategy::ONE_VS_REST);

        REQUIRE(strategy != nullptr);
        REQUIRE(strategy->get_strategy_type() == MulticlassStrategy::ONE_VS_REST);
        REQUIRE_FALSE(strategy->get_classes().empty() == false); // Not trained yet
        REQUIRE(strategy->get_n_classes() == 0);
    }

    SECTION("Create One-vs-One strategy")
    {
        auto strategy = create_multiclass_strategy(MulticlassStrategy::ONE_VS_ONE);

        REQUIRE(strategy != nullptr);
        REQUIRE(strategy->get_strategy_type() == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE(strategy->get_n_classes() == 0);
    }
}

TEST_CASE("OneVsRestStrategy Basic Functionality", "[unit][multiclass_strategy]")
{
    OneVsRestStrategy strategy;
    DataConverter converter;
    KernelParameters params;

    SECTION("Initial state")
    {
        REQUIRE(strategy.get_strategy_type() == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(strategy.get_n_classes() == 0);
        REQUIRE(strategy.get_classes().empty());
        REQUIRE_FALSE(strategy.supports_probability());
    }

    SECTION("Training with linear kernel")
    {
        auto [X, y] = generate_multiclass_data(60, 3, 3);

        params.set_kernel_type(KernelType::LINEAR);
        params.set_C(1.0);

        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(metrics.training_time >= 0.0);
        REQUIRE(strategy.get_n_classes() == 3);

        auto classes = strategy.get_classes();
        REQUIRE(classes.size() == 3);
        REQUIRE(std::is_sorted(classes.begin(), classes.end()));
    }

    SECTION("Training with RBF kernel")
    {
        auto [X, y] = generate_multiclass_data(50, 2, 2);

        params.set_kernel_type(KernelType::RBF);
        params.set_C(1.0);
        params.set_gamma(0.1);

        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(strategy.get_n_classes() == 2);
    }
}

TEST_CASE("OneVsRestStrategy Prediction", "[unit][multiclass_strategy]")
{
    OneVsRestStrategy strategy;
    DataConverter converter;
    KernelParameters params;

    auto [X, y] = generate_multiclass_data(80, 3, 3);

    // Split data
    auto X_train = X.slice(0, 0, 60);
    auto y_train = y.slice(0, 0, 60);
    auto X_test = X.slice(0, 60);
    auto y_test = y.slice(0, 60);

    params.set_kernel_type(KernelType::LINEAR);
    strategy.fit(X_train, y_train, params, converter);

    SECTION("Basic prediction")
    {
        auto predictions = strategy.predict(X_test, converter);

        REQUIRE(static_cast<int64_t>(predictions.size()) == X_test.size(0));

        // Check that all predictions are valid class labels
        auto classes = strategy.get_classes();
        for (int pred : predictions) {
            REQUIRE(std::find(classes.begin(), classes.end(), pred) != classes.end());
        }
    }

    SECTION("Decision function")
    {
        auto decision_values = strategy.decision_function(X_test, converter);

        REQUIRE(static_cast<int64_t>(decision_values.size()) == X_test.size(0));
        REQUIRE(static_cast<int>(decision_values[0].size()) == strategy.get_n_classes());

        // Decision values should be real numbers
        for (const auto& sample_decisions : decision_values) {
            for (double value : sample_decisions) {
                REQUIRE(std::isfinite(value));
            }
        }
    }

    SECTION("Prediction without training")
    {
        OneVsRestStrategy untrained_strategy;

        REQUIRE_THROWS_AS(untrained_strategy.predict(X_test, converter), std::runtime_error);
    }
}

TEST_CASE("OneVsRestStrategy Probability Prediction", "[unit][multiclass_strategy]")
{
    OneVsRestStrategy strategy;
    DataConverter converter;
    KernelParameters params;

    auto [X, y] = generate_multiclass_data(60, 2, 3);

    SECTION("With probability enabled")
    {
        params.set_kernel_type(KernelType::RBF);
        params.set_probability(true);

        strategy.fit(X, y, params, converter);

        if (strategy.supports_probability()) {
            auto probabilities = strategy.predict_proba(X, converter);

            REQUIRE(static_cast<int64_t>(probabilities.size()) == X.size(0));
            REQUIRE(probabilities[0].size() == 3);  // 3 classes

            // Check probability constraints
            for (const auto& sample_probs : probabilities) {
                double sum = 0.0;
                for (double prob : sample_probs) {
                    REQUIRE(prob >= 0.0);
                    REQUIRE(prob <= 1.0);
                    sum += prob;
                }
                REQUIRE(sum == Catch::Approx(1.0).margin(1e-6));
            }
        }
    }

    SECTION("Without probability enabled")
    {
        params.set_kernel_type(KernelType::LINEAR);
        params.set_probability(false);

        strategy.fit(X, y, params, converter);

        // May or may not support probability depending on implementation
        // If not supported, should throw
        if (!strategy.supports_probability()) {
            REQUIRE_THROWS_AS(strategy.predict_proba(X, converter), std::runtime_error);
        }
    }
}

TEST_CASE("OneVsOneStrategy Basic Functionality", "[unit][multiclass_strategy]")
{
    OneVsOneStrategy strategy;
    DataConverter converter;
    KernelParameters params;

    SECTION("Initial state")
    {
        REQUIRE(strategy.get_strategy_type() == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE(strategy.get_n_classes() == 0);
        REQUIRE(strategy.get_classes().empty());
    }

    SECTION("Training with multiple classes")
    {
        auto [X, y] = generate_multiclass_data(80, 3, 4);  // 4 classes for OvO

        params.set_kernel_type(KernelType::LINEAR);
        params.set_C(1.0);

        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(strategy.get_n_classes() == 4);

        auto classes = strategy.get_classes();
        REQUIRE(classes.size() == 4);

        // For 4 classes, OvO should train C(4,2) = 6 binary classifiers
        // This is implementation detail but good to verify the concept
    }

    SECTION("Binary classification")
    {
        auto [X, y] = generate_multiclass_data(50, 2, 2);

        params.set_kernel_type(KernelType::RBF);
        params.set_gamma(0.1);

        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(strategy.get_n_classes() == 2);
    }
}

TEST_CASE("OneVsOneStrategy Prediction", "[unit][multiclass_strategy]")
{
    OneVsOneStrategy strategy;
    DataConverter converter;
    KernelParameters params;

    auto [X, y] = generate_multiclass_data(90, 2, 3);

    auto X_train = X.slice(0, 0, 70);
    auto y_train = y.slice(0, 0, 70);
    auto X_test = X.slice(0, 70);

    params.set_kernel_type(KernelType::LINEAR);
    strategy.fit(X_train, y_train, params, converter);

    SECTION("Basic prediction")
    {
        auto predictions = strategy.predict(X_test, converter);

        REQUIRE(static_cast<int64_t>(predictions.size()) == X_test.size(0));

        auto classes = strategy.get_classes();
        for (int pred : predictions) {
            REQUIRE(std::find(classes.begin(), classes.end(), pred) != classes.end());
        }
    }

    SECTION("Decision function")
    {
        auto decision_values = strategy.decision_function(X_test, converter);

        REQUIRE(static_cast<int64_t>(decision_values.size()) == X_test.size(0));

        // For 3 classes, OvO should have C(3,2) = 3 pairwise comparisons
        REQUIRE(decision_values[0].size() == 3);

        for (const auto& sample_decisions : decision_values) {
            for (double value : sample_decisions) {
                REQUIRE(std::isfinite(value));
            }
        }
    }

    SECTION("Probability prediction")
    {
        // OvO probability estimation is more complex
        auto probabilities = strategy.predict_proba(X_test, converter);

        REQUIRE(static_cast<int64_t>(probabilities.size()) == X_test.size(0));
        REQUIRE(probabilities[0].size() == 3);  // 3 classes

        // Check basic probability constraints
        for (const auto& sample_probs : probabilities) {
            double sum = 0.0;
            for (double prob : sample_probs) {
                REQUIRE(prob >= 0.0);
                REQUIRE(prob <= 1.0);
                sum += prob;
            }
            // OvO probability might not sum exactly to 1 due to voting mechanism
            REQUIRE(sum == Catch::Approx(1.0).margin(0.1));
        }
    }
}

TEST_CASE("MulticlassStrategy Comparison", "[integration][multiclass_strategy]")
{
    auto [X, y] = generate_multiclass_data(100, 3, 3);

    auto X_train = X.slice(0, 0, 80);
    auto y_train = y.slice(0, 0, 80);
    auto X_test = X.slice(0, 80);
    auto y_test = y.slice(0, 80);

    DataConverter converter1, converter2;
    KernelParameters params;
    params.set_kernel_type(KernelType::LINEAR);
    params.set_C(1.0);

    SECTION("Compare OvR vs OvO predictions")
    {
        OneVsRestStrategy ovr_strategy;
        OneVsOneStrategy ovo_strategy;

        ovr_strategy.fit(X_train, y_train, params, converter1);
        ovo_strategy.fit(X_train, y_train, params, converter2);

        auto ovr_predictions = ovr_strategy.predict(X_test, converter1);
        auto ovo_predictions = ovo_strategy.predict(X_test, converter2);

        REQUIRE(ovr_predictions.size() == ovo_predictions.size());

        // Both should predict valid class labels
        auto ovr_classes = ovr_strategy.get_classes();
        auto ovo_classes = ovo_strategy.get_classes();

        REQUIRE(ovr_classes == ovo_classes);  // Should have same classes

        for (size_t i = 0; i < ovr_predictions.size(); ++i) {
            REQUIRE(std::find(ovr_classes.begin(), ovr_classes.end(), ovr_predictions[i]) != ovr_classes.end());
            REQUIRE(std::find(ovo_classes.begin(), ovo_classes.end(), ovo_predictions[i]) != ovo_classes.end());
        }
    }

    SECTION("Compare decision function outputs")
    {
        OneVsRestStrategy ovr_strategy;
        OneVsOneStrategy ovo_strategy;

        ovr_strategy.fit(X_train, y_train, params, converter1);
        ovo_strategy.fit(X_train, y_train, params, converter2);

        auto ovr_decisions = ovr_strategy.decision_function(X_test, converter1);
        auto ovo_decisions = ovo_strategy.decision_function(X_test, converter2);

        REQUIRE(ovr_decisions.size() == ovo_decisions.size());

        // OvR should have one decision value per class
        REQUIRE(ovr_decisions[0].size() == 3);

        // OvO should have one decision value per class pair: C(3,2) = 3
        REQUIRE(ovo_decisions[0].size() == 3);
    }
}

TEST_CASE("MulticlassStrategy Edge Cases", "[unit][multiclass_strategy]")
{
    DataConverter converter;
    KernelParameters params;
    params.set_kernel_type(KernelType::LINEAR);

    SECTION("Single class dataset")
    {
        auto X = torch::randn({ 20, 2 });
        auto y = torch::zeros({ 20 }, torch::kInt32);  // All same class

        OneVsRestStrategy strategy;

        // Should handle single class gracefully
        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        // Implementation might extend to binary case

        auto predictions = strategy.predict(X, converter);
        REQUIRE(static_cast<int64_t>(predictions.size()) == X.size(0));
    }

    SECTION("Very small dataset")
    {
        auto X = torch::tensor({ {1.0, 2.0}, {3.0, 4.0} });
        auto y = torch::tensor({ 0, 1 });

        OneVsOneStrategy strategy;

        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);

        auto predictions = strategy.predict(X, converter);
        REQUIRE(predictions.size() == 2);
    }

    SECTION("Imbalanced classes")
    {
        // Create dataset with very imbalanced classes
        auto X1 = torch::randn({ 80, 2 });
        auto y1 = torch::zeros({ 80 }, torch::kInt32);

        auto X2 = torch::randn({ 5, 2 });
        auto y2 = torch::ones({ 5 }, torch::kInt32);

        auto X = torch::cat({ X1, X2 }, 0);
        auto y = torch::cat({ y1, y2 }, 0);

        OneVsRestStrategy strategy;
        auto metrics = strategy.fit(X, y, params, converter);

        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(strategy.get_n_classes() == 2);

        auto predictions = strategy.predict(X, converter);
        REQUIRE(static_cast<int64_t>(predictions.size()) == X.size(0));
    }
}

TEST_CASE("MulticlassStrategy Error Handling", "[unit][multiclass_strategy]")
{
    DataConverter converter;
    KernelParameters params;

    SECTION("Invalid parameters")
    {
        OneVsRestStrategy strategy;
        auto [X, y] = generate_multiclass_data(50, 2, 2);

        // Invalid C parameter
        params.set_kernel_type(KernelType::LINEAR);
        params.set_C(-1.0);  // Invalid

        REQUIRE_THROWS(strategy.fit(X, y, params, converter));
    }

    SECTION("Mismatched tensor dimensions")
    {
        OneVsOneStrategy strategy;

        auto X = torch::randn({ 50, 3 });
        auto y = torch::randint(0, 2, { 40 });  // Wrong number of labels

        params.set_kernel_type(KernelType::LINEAR);
        params.set_C(1.0);

        REQUIRE_THROWS_AS(strategy.fit(X, y, params, converter), std::invalid_argument);
    }

    SECTION("Prediction on untrained strategy")
    {
        OneVsRestStrategy strategy;
        auto X = torch::randn({ 10, 2 });

        REQUIRE_THROWS_AS(strategy.predict(X, converter), std::runtime_error);
        REQUIRE_THROWS_AS(strategy.decision_function(X, converter), std::runtime_error);
    }
}

TEST_CASE("MulticlassStrategy Memory Management", "[unit][multiclass_strategy]")
{
    SECTION("Strategy destruction")
    {
        // Test that strategies clean up properly
        auto strategy = create_multiclass_strategy(MulticlassStrategy::ONE_VS_REST);

        DataConverter converter;
        KernelParameters params;
        auto [X, y] = generate_multiclass_data(50, 2, 3);

        params.set_kernel_type(KernelType::LINEAR);
        strategy->fit(X, y, params, converter);

        REQUIRE(strategy->get_n_classes() == 3);

        // Strategy should clean up automatically when destroyed
    }

    SECTION("Multiple training rounds")
    {
        OneVsRestStrategy strategy;
        DataConverter converter;
        KernelParameters params;
        params.set_kernel_type(KernelType::LINEAR);

        // Train multiple times with different data
        for (int i = 0; i < 3; ++i) {
            auto [X, y] = generate_multiclass_data(40, 2, 2, i);  // Different seed

            auto metrics = strategy.fit(X, y, params, converter);
            REQUIRE(metrics.status == TrainingStatus::SUCCESS);

            auto predictions = strategy.predict(X, converter);
            REQUIRE(static_cast<int64_t>(predictions.size()) == X.size(0));
        }
    }
}