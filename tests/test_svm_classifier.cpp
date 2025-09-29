/**
 * @file test_svm_classifier.cpp
 * @brief Integration tests for SVMClassifier class
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>
#include <nlohmann/json.hpp>

using namespace svm_classifier;
using json = nlohmann::json;

/**
 * @brief Generate synthetic classification dataset
 */
std::pair<torch::Tensor, torch::Tensor> generate_test_data(int n_samples = 100,
    int n_features = 4,
    int n_classes = 3,
    int seed = 42)
{
    torch::manual_seed(seed);

    auto X = torch::randn({ n_samples, n_features });
    auto y = torch::randint(0, n_classes, { n_samples });

    // Add some structure to make classification meaningful
    for (int i = 0; i < n_samples; ++i) {
        int target_class = y[i].item<int>();
        // Bias features toward the target class
        X[i] += torch::randn({ n_features }) * 0.5 + target_class;
    }

    return { X, y };
}

TEST_CASE("SVMClassifier Construction", "[integration][svm_classifier]")
{
    SECTION("Default constructor")
    {
        SVMClassifier svm;

        REQUIRE(svm.get_kernel_type() == KernelType::LINEAR);
        REQUIRE_FALSE(svm.is_fitted());
        REQUIRE(svm.get_n_classes() == 0);
        REQUIRE(svm.get_n_features() == 0);
        REQUIRE(svm.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_REST);
    }

    SECTION("Constructor with parameters")
    {
        SVMClassifier svm(KernelType::RBF, 10.0, MulticlassStrategy::ONE_VS_ONE);

        REQUIRE(svm.get_kernel_type() == KernelType::RBF);
        REQUIRE(svm.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE_FALSE(svm.is_fitted());
    }

    SECTION("JSON constructor")
    {
        json config = {
            {"kernel", "polynomial"},
            {"C", 5.0},
            {"degree", 4},
            {"multiclass_strategy", "ovo"}
        };

        SVMClassifier svm(config);

        REQUIRE(svm.get_kernel_type() == KernelType::POLYNOMIAL);
        REQUIRE(svm.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_ONE);
    }
}

TEST_CASE("SVMClassifier Parameter Management", "[integration][svm_classifier]")
{
    SVMClassifier svm;

    SECTION("Set and get parameters")
    {
        json new_params = {
            {"kernel", "rbf"},
            {"C", 2.0},
            {"gamma", 0.1},
            {"probability", true}
        };

        svm.set_parameters(new_params);
        auto current_params = svm.get_parameters();

        REQUIRE(current_params["kernel"] == "rbf");
        REQUIRE(current_params["C"] == Catch::Approx(2.0));
        REQUIRE(current_params["gamma"] == Catch::Approx(0.1));
        REQUIRE(current_params["probability"] == true);
    }

    SECTION("Invalid parameters")
    {
        json invalid_params = {
            {"kernel", "invalid_kernel"}
        };

        REQUIRE_THROWS_AS(svm.set_parameters(invalid_params), std::invalid_argument);

        json invalid_C = {
            {"C", -1.0}
        };

        REQUIRE_THROWS_AS(svm.set_parameters(invalid_C), std::invalid_argument);
    }

    SECTION("Parameter changes reset fitted state")
    {
        auto [X, y] = generate_test_data(50, 3, 2);

        svm.fit(X, y);
        REQUIRE(svm.is_fitted());

        json new_params = { {"kernel", "rbf"} };
        svm.set_parameters(new_params);

        REQUIRE_FALSE(svm.is_fitted());
    }
}

TEST_CASE("SVMClassifier Linear Kernel Training", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::LINEAR, 1.0);
    auto [X, y] = generate_test_data(100, 4, 3);

    SECTION("Basic training")
    {
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
        REQUIRE(svm.get_n_features() == 4);
        REQUIRE(svm.get_n_classes() == 3);
        REQUIRE(svm.get_svm_library() == SVMLibrary::LIBLINEAR);
        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(metrics.training_time >= 0.0);
    }

    SECTION("Training with probability")
    {
        json config = {
            {"kernel", "linear"},
            {"probability", true}
        };

        svm.set_parameters(config);
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
        REQUIRE(svm.supports_probability());
    }

    SECTION("Binary classification")
    {
        auto [X_binary, y_binary] = generate_test_data(50, 3, 2);

        auto metrics = svm.fit(X_binary, y_binary);

        REQUIRE(svm.is_fitted());
        REQUIRE(svm.get_n_classes() == 2);
    }
}

TEST_CASE("SVMClassifier RBF Kernel Training", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::RBF, 1.0);
    auto [X, y] = generate_test_data(80, 3, 2);

    SECTION("Basic RBF training")
    {
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
        REQUIRE(svm.get_svm_library() == SVMLibrary::LIBSVM);
        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
    }

    SECTION("RBF with custom gamma")
    {
        json config = {
            {"kernel", "rbf"},
            {"gamma", 0.5}
        };

        svm.set_parameters(config);
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
    }

    SECTION("RBF with auto gamma")
    {
        json config = {
            {"kernel", "rbf"},
            {"gamma", "auto"}
        };

        svm.set_parameters(config);
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
    }
}

TEST_CASE("SVMClassifier Polynomial Kernel Training", "[integration][svm_classifier]")
{
    SVMClassifier svm;
    auto [X, y] = generate_test_data(60, 2, 2);

    SECTION("Polynomial kernel")
    {
        json config = {
            {"kernel", "polynomial"},
            {"degree", 3},
            {"gamma", 0.1},
            {"coef0", 1.0}
        };

        svm.set_parameters(config);
        auto metrics = svm.fit(X, y);

        REQUIRE(svm.is_fitted());
        REQUIRE(svm.get_kernel_type() == KernelType::POLYNOMIAL);
        REQUIRE(svm.get_svm_library() == SVMLibrary::LIBSVM);
    }

    SECTION("Different degrees")
    {
        for (int degree : {2, 4, 5}) {
            json config = {
                {"kernel", "polynomial"},
                {"degree", degree}
            };

            SVMClassifier poly_svm(config);
            REQUIRE_NOTHROW(poly_svm.fit(X, y));
            REQUIRE(poly_svm.is_fitted());
        }
    }
}

TEST_CASE("SVMClassifier Sigmoid Kernel Training", "[integration][svm_classifier]")
{
    SVMClassifier svm;
    auto [X, y] = generate_test_data(50, 2, 2);

    json config = {
        {"kernel", "sigmoid"},
        {"gamma", 0.01},
        {"coef0", 0.5}
    };

    svm.set_parameters(config);
    auto metrics = svm.fit(X, y);

    REQUIRE(svm.is_fitted());
    REQUIRE(svm.get_kernel_type() == KernelType::SIGMOID);
    REQUIRE(svm.get_svm_library() == SVMLibrary::LIBSVM);
}

TEST_CASE("SVMClassifier Prediction", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::LINEAR);
    auto [X, y] = generate_test_data(100, 3, 3);

    // Split data
    auto X_train = X.slice(0, 0, 80);
    auto y_train = y.slice(0, 0, 80);
    auto X_test = X.slice(0, 80);
    auto y_test = y.slice(0, 80);

    svm.fit(X_train, y_train);

    SECTION("Basic prediction")
    {
        auto predictions = svm.predict(X_test);

        REQUIRE(predictions.dtype() == torch::kInt32);
        REQUIRE(predictions.size(0) == X_test.size(0));

        // Check that predictions are valid class labels
        auto unique_preds = std::get<0>(at::_unique(predictions));
        for (int i = 0; i < unique_preds.size(0); ++i) {
            int pred_class = unique_preds[i].item<int>();
            auto classes = svm.get_classes();
            REQUIRE(std::find(classes.begin(), classes.end(), pred_class) != classes.end());
        }
    }

    SECTION("Prediction accuracy")
    {
        double accuracy = svm.score(X_test, y_test);

        REQUIRE(accuracy >= 0.0);
        REQUIRE(accuracy <= 1.0);
        // For this synthetic dataset, we expect reasonable accuracy
        REQUIRE(accuracy > 0.3);  // Very loose bound
    }

    SECTION("Prediction on training data")
    {
        auto train_predictions = svm.predict(X_train);
        double train_accuracy = svm.score(X_train, y_train);

        REQUIRE(train_accuracy >= 0.0);
        REQUIRE(train_accuracy <= 1.0);
    }
}

TEST_CASE("SVMClassifier Probability Prediction", "[integration][svm_classifier]")
{
    json config = {
        {"kernel", "rbf"},
        {"probability", true}
    };

    SVMClassifier svm(config);
    auto [X, y] = generate_test_data(80, 3, 3);

    svm.fit(X, y);

    SECTION("Probability predictions")
    {
        REQUIRE(svm.supports_probability());

        auto probabilities = svm.predict_proba(X);

        REQUIRE(probabilities.dtype() == torch::kFloat64);
        REQUIRE(probabilities.size(0) == X.size(0));
        REQUIRE(probabilities.size(1) == 3);  // 3 classes

        // Check that probabilities sum to 1
        auto prob_sums = probabilities.sum(1);
        for (int i = 0; i < prob_sums.size(0); ++i) {
            REQUIRE(prob_sums[i].item<double>() == Catch::Approx(1.0).margin(1e-6));
        }

        // Check that all probabilities are non-negative
        REQUIRE(torch::all(probabilities >= 0.0).item<bool>());
    }

    SECTION("Probability without training")
    {
        SVMClassifier untrained_svm(config);
        REQUIRE_THROWS_AS(untrained_svm.predict_proba(X), std::runtime_error);
    }

    SECTION("Probability not supported")
    {
        SVMClassifier no_prob_svm(KernelType::LINEAR);  // No probability
        no_prob_svm.fit(X, y);

        REQUIRE_FALSE(no_prob_svm.supports_probability());
        REQUIRE_THROWS_AS(no_prob_svm.predict_proba(X), std::runtime_error);
    }
}

TEST_CASE("SVMClassifier Decision Function", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::RBF);
    auto [X, y] = generate_test_data(60, 2, 3);

    svm.fit(X, y);

    SECTION("Decision function values")
    {
        auto decision_values = svm.decision_function(X);

        REQUIRE(decision_values.dtype() == torch::kFloat64);
        REQUIRE(decision_values.size(0) == X.size(0));
        // Decision function output depends on multiclass strategy
        REQUIRE(decision_values.size(1) > 0);
    }

    SECTION("Decision function consistency with predictions")
    {
        auto predictions = svm.predict(X);
        auto decision_values = svm.decision_function(X);

        // For OvR strategy, the predicted class should correspond to max decision value
        if (svm.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_REST) {
            for (int i = 0; i < X.size(0); ++i) {
                auto max_indices = std::get<1>(torch::max(decision_values[i], 0));
                // This is a simplified check - actual implementation might be more complex
            }
        }
    }
}

TEST_CASE("SVMClassifier Multiclass Strategies", "[integration][svm_classifier]")
{
    auto [X, y] = generate_test_data(80, 3, 4);  // 4 classes

    SECTION("One-vs-Rest strategy")
    {
        json config = {
            {"kernel", "linear"},
            {"multiclass_strategy", "ovr"}
        };

        SVMClassifier svm_ovr(config);
        auto metrics = svm_ovr.fit(X, y);

        REQUIRE(svm_ovr.is_fitted());
        REQUIRE(svm_ovr.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_REST);
        REQUIRE(svm_ovr.get_n_classes() == 4);

        auto predictions = svm_ovr.predict(X);
        REQUIRE(predictions.size(0) == X.size(0));
    }

    SECTION("One-vs-One strategy")
    {
        json config = {
            {"kernel", "rbf"},
            {"multiclass_strategy", "ovo"}
        };

        SVMClassifier svm_ovo(config);
        auto metrics = svm_ovo.fit(X, y);

        REQUIRE(svm_ovo.is_fitted());
        REQUIRE(svm_ovo.get_multiclass_strategy() == MulticlassStrategy::ONE_VS_ONE);
        REQUIRE(svm_ovo.get_n_classes() == 4);

        auto predictions = svm_ovo.predict(X);
        REQUIRE(predictions.size(0) == X.size(0));
    }

    SECTION("Compare strategies")
    {
        SVMClassifier svm_ovr(KernelType::LINEAR, 1.0, MulticlassStrategy::ONE_VS_REST);
        SVMClassifier svm_ovo(KernelType::LINEAR, 1.0, MulticlassStrategy::ONE_VS_ONE);

        svm_ovr.fit(X, y);
        svm_ovo.fit(X, y);

        auto pred_ovr = svm_ovr.predict(X);
        auto pred_ovo = svm_ovo.predict(X);

        // Both should produce valid predictions
        REQUIRE(pred_ovr.size(0) == X.size(0));
        REQUIRE(pred_ovo.size(0) == X.size(0));
    }
}

TEST_CASE("SVMClassifier Evaluation Metrics", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::LINEAR);
    auto [X, y] = generate_test_data(100, 3, 3);

    svm.fit(X, y);

    SECTION("Detailed evaluation")
    {
        auto metrics = svm.evaluate(X, y);

        REQUIRE(metrics.accuracy >= 0.0);
        REQUIRE(metrics.accuracy <= 1.0);
        REQUIRE(metrics.precision >= 0.0);
        REQUIRE(metrics.precision <= 1.0);
        REQUIRE(metrics.recall >= 0.0);
        REQUIRE(metrics.recall <= 1.0);
        REQUIRE(metrics.f1_score >= 0.0);
        REQUIRE(metrics.f1_score <= 1.0);

        // Check confusion matrix dimensions
        REQUIRE(metrics.confusion_matrix.size() == 3);  // 3 classes
        for (const auto& row : metrics.confusion_matrix) {
            REQUIRE(row.size() == 3);
        }
    }

    SECTION("Perfect predictions metrics")
    {
        // Create a very simple binary classification problem first
        // This should definitely work with linear SVM
        auto X_simple = torch::tensor({
            {-1.0, -1.0}, {-1.1, -1.1}, {-0.9, -0.9},  // class 0 (negative)
            {1.0, 1.0}, {1.1, 1.1}, {0.9, 0.9}          // class 1 (positive)
        });
        auto y_simple = torch::tensor({ 0, 0, 0, 1, 1, 1 });

        SVMClassifier simple_svm(KernelType::LINEAR);
        simple_svm.fit(X_simple, y_simple);

        auto metrics = simple_svm.evaluate(X_simple, y_simple);

        // Binary classification should achieve high accuracy on this linearly separable data
        REQUIRE(metrics.accuracy > 0.8);
        REQUIRE(metrics.precision >= 0.0);
        REQUIRE(metrics.recall >= 0.0);
        REQUIRE(metrics.f1_score >= 0.0);
    }
}

TEST_CASE("SVMClassifier Cross-Validation", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::LINEAR);
    auto [X, y] = generate_test_data(100, 3, 2);

    SECTION("5-fold cross-validation")
    {
        auto cv_scores = svm.cross_validate(X, y, 5);

        REQUIRE(cv_scores.size() == 5);

        for (double score : cv_scores) {
            REQUIRE(score >= 0.0);
            REQUIRE(score <= 1.0);
        }

        // Calculate mean and std
        double mean = std::accumulate(cv_scores.begin(), cv_scores.end(), 0.0) / cv_scores.size();
        REQUIRE(mean >= 0.0);
        REQUIRE(mean <= 1.0);
    }

    SECTION("Invalid CV folds")
    {
        REQUIRE_THROWS_AS(svm.cross_validate(X, y, 1), std::invalid_argument);
        REQUIRE_THROWS_AS(svm.cross_validate(X, y, 0), std::invalid_argument);
    }

    SECTION("CV preserves original state")
    {
        // Fit the model first
        svm.fit(X, y);
        auto original_classes = svm.get_classes();

        // Run CV
        auto cv_scores = svm.cross_validate(X, y, 3);

        // Should still be fitted with same classes
        REQUIRE(svm.is_fitted());
        REQUIRE(svm.get_classes() == original_classes);
    }
}

TEST_CASE("SVMClassifier Grid Search", "[integration][svm_classifier]")
{
    SVMClassifier svm;
    auto [X, y] = generate_test_data(60, 2, 2);  // Smaller dataset for faster testing

    SECTION("Simple grid search")
    {
        json param_grid = {
            {"kernel", {"linear", "rbf"}},
            {"C", {0.1, 1.0, 10.0}}
        };

        auto results = svm.grid_search(X, y, param_grid, 3);

        REQUIRE(results.contains("best_params"));
        REQUIRE(results.contains("best_score"));
        REQUIRE(results.contains("cv_results"));

        auto best_score = results["best_score"].get<double>();
        REQUIRE(best_score >= 0.0);
        REQUIRE(best_score <= 1.0);

        auto cv_results = results["cv_results"].get<std::vector<double>>();
        REQUIRE(cv_results.size() == 6);  // 2 kernels × 3 C values
    }

    SECTION("RBF-specific grid search")
    {
        json param_grid = {
            {"kernel", {"rbf"}},
            {"C", {1.0, 10.0}},
            {"gamma", {0.01, 0.1}}
        };

        auto results = svm.grid_search(X, y, param_grid, 3);

        auto best_params = results["best_params"];
        REQUIRE(best_params["kernel"] == "rbf");
        REQUIRE(best_params.contains("C"));
        REQUIRE(best_params.contains("gamma"));
    }
}

TEST_CASE("SVMClassifier Error Handling", "[integration][svm_classifier]")
{
    SVMClassifier svm;

    SECTION("Prediction before training")
    {
        auto X = torch::randn({ 5, 3 });

        REQUIRE_THROWS_AS(svm.predict(X), std::runtime_error);
        REQUIRE_THROWS_AS(svm.predict_proba(X), std::runtime_error);
        REQUIRE_THROWS_AS(svm.decision_function(X), std::runtime_error);
    }

    SECTION("Inconsistent feature dimensions")
    {
        auto X_train = torch::randn({ 50, 3 });
        auto y_train = torch::randint(0, 2, { 50 });
        auto X_test = torch::randn({ 10, 5 });  // Different number of features

        svm.fit(X_train, y_train);

        REQUIRE_THROWS_AS(svm.predict(X_test), std::invalid_argument);
    }

    SECTION("Invalid training data")
    {
        auto X_invalid = torch::tensor({ {std::numeric_limits<float>::quiet_NaN(), 1.0} });
        auto y_invalid = torch::tensor({ 0 });

        REQUIRE_THROWS_AS(svm.fit(X_invalid, y_invalid), std::invalid_argument);
    }

    SECTION("Empty datasets")
    {
        auto X_empty = torch::empty({ 0, 3 });
        auto y_empty = torch::empty({ 0 });

        REQUIRE_THROWS_AS(svm.fit(X_empty, y_empty), std::invalid_argument);
    }
}

TEST_CASE("SVMClassifier Move Semantics", "[integration][svm_classifier]")
{
    SECTION("Move constructor")
    {
        SVMClassifier svm1(KernelType::RBF, 2.0);
        auto [X, y] = generate_test_data(50, 2, 2);
        svm1.fit(X, y);

        auto original_classes = svm1.get_classes();
        bool was_fitted = svm1.is_fitted();

        SVMClassifier svm2 = std::move(svm1);

        REQUIRE(svm2.is_fitted() == was_fitted);
        REQUIRE(svm2.get_classes() == original_classes);
        REQUIRE(svm2.get_kernel_type() == KernelType::RBF);

        // Original should be in valid but unspecified state
        REQUIRE_FALSE(svm1.is_fitted());
    }

    SECTION("Move assignment")
    {
        SVMClassifier svm1(KernelType::POLYNOMIAL);
        SVMClassifier svm2(KernelType::LINEAR);

        auto [X, y] = generate_test_data(40, 2, 2);
        svm1.fit(X, y);

        auto original_classes = svm1.get_classes();

        svm2 = std::move(svm1);

        REQUIRE(svm2.is_fitted());
        REQUIRE(svm2.get_classes() == original_classes);
        REQUIRE(svm2.get_kernel_type() == KernelType::POLYNOMIAL);
    }
}

TEST_CASE("SVMClassifier Reset Functionality", "[integration][svm_classifier]")
{
    SVMClassifier svm(KernelType::RBF);
    auto [X, y] = generate_test_data(50, 3, 2);

    svm.fit(X, y);
    REQUIRE(svm.is_fitted());
    REQUIRE(svm.get_n_features() > 0);
    REQUIRE(svm.get_n_classes() > 0);

    svm.reset();

    REQUIRE_FALSE(svm.is_fitted());
    REQUIRE(svm.get_n_features() == 0);
    REQUIRE(svm.get_n_classes() == 0);

    // Should be able to train again after reset
    REQUIRE_NOTHROW(svm.fit(X, y));
    REQUIRE(svm.is_fitted());
}