# SVM Classifier - Complete Guide

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Hyperparameters Reference](#hyperparameters-reference)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

---

## Overview

SVMClassifier is a high-performance Support Vector Machine library for C++ with a scikit-learn compatible API. It provides a unified interface that automatically selects the optimal underlying implementation based on the kernel type:

- **Linear kernels** → liblinear (highly optimized for linear classification)
- **Non-linear kernels** (RBF, Polynomial, Sigmoid) → libsvm (supports arbitrary kernels)

The library supports multiclass classification through One-vs-Rest and One-vs-One strategies and integrates seamlessly with PyTorch tensors.

---

## How It Works

### Architecture

The SVM classifier follows a modular design with the following key components:

#### 1. **SVMClassifier** (Main Interface)
   - Entry point for all SVM operations
   - Manages model lifecycle (fit, predict, evaluate)
   - Handles parameter configuration
   - Provides scikit-learn compatible API

**Location**: `include/svm_classifier/svm_classifier.hpp`, `src/svm_classifier.cpp`

#### 2. **KernelParameters** (Parameter Management)
   - Stores and validates all hyperparameters
   - Handles kernel-specific parameter requirements
   - Provides parameter serialization to/from JSON
   - Validates parameter consistency

**Location**: `include/svm_classifier/kernel_parameters.hpp`, `src/kernel_parameters.cpp`

#### 3. **DataConverter** (Data Processing)
   - Converts PyTorch tensors to libsvm/liblinear formats
   - Handles both dense and sparse data representations
   - Manages memory efficiently for large datasets
   - Provides automatic gamma calculation

**Location**: `include/svm_classifier/data_converter.hpp`, `src/data_converter.cpp`

#### 4. **MulticlassStrategy** (Multiclass Handling)
   - Implements One-vs-Rest (OvR) strategy
   - Implements One-vs-One (OvO) strategy
   - Aggregates predictions from binary classifiers
   - Computes probability estimates

**Location**: `include/svm_classifier/multiclass_strategy.hpp`, `src/multiclass_strategy.cpp`

### Automatic Library Selection

The classifier intelligently selects the optimal backend:

```cpp
inline SVMLibrary get_svm_library(KernelType kernel) {
    return (kernel == KernelType::LINEAR) ? SVMLibrary::LIBLINEAR : SVMLibrary::LIBSVM;
}
```

- **liblinear**: Optimized for linear SVMs, faster training on high-dimensional data
- **libsvm**: General-purpose SVM supporting all kernel types

### Workflow

1. **Initialization**: Create classifier with desired parameters
2. **Training**: Call `fit(X, y)` with feature and label tensors
   - Data is validated and converted to appropriate format
   - Multiclass strategy creates binary classifiers
   - Each classifier is trained using liblinear or libsvm
   - Training metrics are collected
3. **Prediction**: Call `predict(X)` for class labels
   - Features are converted to appropriate format
   - Each binary classifier makes predictions
   - Multiclass strategy aggregates results
4. **Evaluation**: Use `score()`, `evaluate()`, or `predict_proba()`

---

## Hyperparameters Reference

### Core Parameters

#### `kernel` (string or KernelType enum)
**Description**: Specifies the kernel function used to transform data into higher-dimensional space.

**Options**:
- `"linear"` or `KernelType::LINEAR`: Linear kernel `<x, y>`
  - Best for: High-dimensional data, text classification
  - Computational complexity: O(n × m) where n=samples, m=features

- `"rbf"` or `KernelType::RBF`: Radial Basis Function `exp(-gamma * ||x - y||²)`
  - Best for: General-purpose non-linear classification
  - Most popular choice for non-linear problems

- `"polynomial"` or `KernelType::POLYNOMIAL`: Polynomial `(gamma * <x, y> + coef0)^degree`
  - Best for: Problems with natural polynomial relationships
  - Can be unstable with high degrees

- `"sigmoid"` or `KernelType::SIGMOID`: Sigmoid/tanh `tanh(gamma * <x, y> + coef0)`
  - Best for: Neural network-like behavior
  - Less commonly used

**Default**: `"linear"`

**Example**:
```cpp
// Using string
json config = {{"kernel", "rbf"}};
SVMClassifier svm(config);

// Using enum
SVMClassifier svm(KernelType::RBF, 1.0);
```

---

#### `C` (double)
**Description**: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification error.

**Range**: C > 0 (must be positive)

**Effect**:
- **Low C** (e.g., 0.01-0.1): Soft margin, prioritizes larger margin over correct classification
  - More regularization
  - Better generalization on noisy data
  - May underfit

- **High C** (e.g., 10-1000): Hard margin, prioritizes correct classification
  - Less regularization
  - Fits training data more closely
  - May overfit

**Default**: `1.0`

**Usage Guidelines**:
- Start with C=1.0 and adjust based on validation performance
- Increase C if model is underfitting
- Decrease C if model is overfitting
- Use cross-validation to find optimal value
- Typical search range: [0.01, 0.1, 1.0, 10.0, 100.0]

**Example**:
```cpp
json config = {{"kernel", "linear"}, {"C", 10.0}};
```

---

#### `multiclass_strategy` (string or MulticlassStrategy enum)
**Description**: Strategy for handling multiclass classification problems.

**Options**:
- `"ovr"` or `"one_vs_rest"` or `MulticlassStrategy::ONE_VS_REST`
  - Trains N binary classifiers (one per class)
  - Each classifier: "class i" vs "all other classes"
  - Prediction: class with highest confidence
  - Faster training for many classes
  - More interpretable

- `"ovo"` or `"one_vs_one"` or `MulticlassStrategy::ONE_VS_ONE`
  - Trains N×(N-1)/2 binary classifiers (one per pair)
  - Each classifier: "class i" vs "class j"
  - Prediction: majority voting
  - Better for imbalanced datasets
  - More robust but slower for many classes

**Default**: `"ovr"` (One-vs-Rest)

**Comparison**:
| Aspect | OvR | OvO |
|--------|-----|-----|
| # Classifiers | N | N×(N-1)/2 |
| Training time | Faster | Slower |
| Prediction time | Faster | Slower |
| Imbalanced data | May struggle | Better handling |
| Memory usage | Lower | Higher |

**Example**:
```cpp
json config = {
    {"kernel", "rbf"},
    {"multiclass_strategy", "ovo"}
};
```

---

### Kernel-Specific Parameters

#### `gamma` (double or string "auto")
**Description**: Kernel coefficient for RBF, polynomial, and sigmoid kernels. Controls the influence of individual training samples.

**Applicable to**: RBF, Polynomial, Sigmoid kernels (ignored for linear)

**Range**:
- gamma > 0 (explicit value)
- `"auto"` or `-1.0` for automatic calculation: `1 / n_features`

**Effect**:
- **Low gamma** (e.g., 0.001-0.01): Wide influence, smoother decision boundary
  - Less complex model
  - May underfit

- **High gamma** (e.g., 1.0-10.0): Narrow influence, more complex decision boundary
  - Points far from decision boundary have little effect
  - May overfit

**Default**: `"auto"` (automatically computed as 1/n_features)

**Usage Guidelines**:
- For RBF kernel: gamma is critical for performance
- Start with "auto" and adjust if needed
- Typical search range: [0.001, 0.01, 0.1, 1.0]
- Should be tuned together with C
- Feature scaling is crucial when using explicit gamma values

**Example**:
```cpp
// Auto gamma
json config = {{"kernel", "rbf"}, {"gamma", "auto"}};

// Explicit gamma
json config = {{"kernel", "rbf"}, {"gamma", 0.1}};
```

---

#### `degree` (int)
**Description**: Degree of the polynomial kernel function.

**Applicable to**: Polynomial kernel only

**Range**: degree ≥ 1 (must be positive integer)

**Effect**:
- **Low degree** (2-3): Simpler polynomial relationships, smoother boundary
- **High degree** (4+): Complex polynomial relationships, risk of overfitting

**Default**: `3`

**Usage Guidelines**:
- Degree 2-3 works well for most problems
- Higher degrees (>5) rarely improve performance
- Computational cost increases with degree
- Higher degrees more prone to numerical instability

**Example**:
```cpp
json config = {
    {"kernel", "polynomial"},
    {"degree", 3},
    {"C", 1.0},
    {"gamma", 0.1}
};
```

---

#### `coef0` (double)
**Description**: Independent term in polynomial and sigmoid kernels.

**Applicable to**: Polynomial and sigmoid kernels

**Range**: Any real number

**Effect**:
- Shifts the kernel function
- In polynomial: affects the balance between lower and higher degree terms
- In sigmoid: affects the offset of the tanh function

**Default**: `0.0`

**Usage Guidelines**:
- Usually left at default (0.0)
- Can try values in range [-1.0, 1.0]
- Has less impact than gamma or C

**Example**:
```cpp
json config = {
    {"kernel", "polynomial"},
    {"degree", 3},
    {"coef0", 0.5}
};
```

---

### Optimization Parameters

#### `tolerance` (double)
**Description**: Tolerance for stopping criterion in the optimization algorithm.

**Range**: tolerance > 0

**Effect**:
- **Lower tolerance** (e.g., 1e-5): More precise optimization, longer training
- **Higher tolerance** (e.g., 1e-2): Faster training, potentially less optimal solution

**Default**: `1e-3` (0.001)

**Usage Guidelines**:
- Default value works well for most cases
- Decrease for very precise models
- Increase to speed up training on large datasets

**Example**:
```cpp
json config = {{"tolerance", 1e-4}};
```

---

#### `max_iterations` (int)
**Description**: Maximum number of iterations for the optimization solver.

**Range**:
- max_iterations > 0 (positive integer)
- `-1` for no limit

**Effect**:
- Prevents infinite loops in optimization
- Training stops when either tolerance or max_iterations is reached

**Default**: `-1` (no limit)

**Usage Guidelines**:
- Use default (-1) for most cases
- Set explicit limit for very large datasets to control training time
- If training doesn't converge, increase max_iterations or adjust other parameters

**Example**:
```cpp
json config = {{"max_iterations", 10000}};
```

---

#### `cache_size` (double)
**Description**: Size of kernel cache in megabytes (for libsvm only).

**Range**: cache_size ≥ 0

**Effect**:
- **Larger cache**: Faster training, more memory usage
- **Smaller cache**: Slower training, less memory usage
- No effect for linear kernels (uses liblinear)

**Default**: `200.0` MB

**Usage Guidelines**:
- Increase for large datasets to speed up training
- Decrease if memory is limited
- No benefit beyond dataset size

**Example**:
```cpp
json config = {{"cache_size", 500.0}};  // 500 MB cache
```

---

### Probability Parameters

#### `probability` (bool)
**Description**: Enable probability estimates for predictions.

**Range**: true or false

**Effect**:
- `true`: Enables `predict_proba()` method, uses Platt scaling for calibration
- `false`: Only decision values available, faster training

**Default**: `false`

**Usage Guidelines**:
- Enable when you need probability estimates
- Slightly increases training time (requires cross-validation)
- Required for applications needing confidence scores

**Example**:
```cpp
json config = {{"probability", true}};

SVMClassifier svm(config);
svm.fit(X_train, y_train);

// Now you can get probabilities
auto proba = svm.predict_proba(X_test);  // Returns probability matrix
```

---

## Usage Examples

### Basic Usage

#### 1. Simple Linear SVM
```cpp
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>

using namespace svm_classifier;

// Create training data
auto X_train = torch::randn({100, 10});  // 100 samples, 10 features
auto y_train = torch::randint(0, 2, {100}, torch::kInt);  // Binary labels

// Create and train classifier with defaults
SVMClassifier svm;  // Uses linear kernel by default
svm.fit(X_train, y_train);

// Make predictions
auto predictions = svm.predict(X_test);

// Get accuracy
double accuracy = svm.score(X_test, y_test);
```

#### 2. Using Different Kernels
```cpp
// RBF kernel
SVMClassifier svm_rbf(KernelType::RBF, 1.0);
svm_rbf.fit(X_train, y_train);

// Polynomial kernel
SVMClassifier svm_poly(KernelType::POLYNOMIAL, 1.0);
svm_poly.fit(X_train, y_train);

// Sigmoid kernel
SVMClassifier svm_sigmoid(KernelType::SIGMOID, 1.0);
svm_sigmoid.fit(X_train, y_train);
```

### JSON Configuration

#### 1. Complete Configuration
```cpp
#include <nlohmann/json.hpp>

using json = nlohmann::json;

json config = {
    // Kernel configuration
    {"kernel", "rbf"},
    {"gamma", 0.1},

    // Regularization
    {"C", 10.0},

    // Multiclass strategy
    {"multiclass_strategy", "ovo"},

    // Optimization
    {"tolerance", 1e-4},
    {"max_iterations", 10000},
    {"cache_size", 500.0},

    // Probability
    {"probability", true}
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);
```

#### 2. Kernel-Specific Configurations
```cpp
// RBF with auto gamma
json rbf_config = {
    {"kernel", "rbf"},
    {"C", 10.0},
    {"gamma", "auto"}
};

// Polynomial with custom degree
json poly_config = {
    {"kernel", "polynomial"},
    {"degree", 4},
    {"gamma", 0.1},
    {"coef0", 1.0},
    {"C", 1.0}
};

// Linear with high regularization
json linear_config = {
    {"kernel", "linear"},
    {"C", 0.1}
};
```

### Advanced Usage

#### 1. Hyperparameter Tuning
```cpp
// Manual grid search
std::vector<double> c_values = {0.1, 1.0, 10.0, 100.0};
std::vector<double> gamma_values = {0.001, 0.01, 0.1, 1.0};

double best_score = 0.0;
json best_params;

for (double c : c_values) {
    for (double gamma : gamma_values) {
        json config = {
            {"kernel", "rbf"},
            {"C", c},
            {"gamma", gamma}
        };

        SVMClassifier svm(config);
        auto cv_scores = svm.cross_validate(X, y, 5);  // 5-fold CV

        double mean_score = std::accumulate(cv_scores.begin(),
                                           cv_scores.end(), 0.0) / cv_scores.size();

        if (mean_score > best_score) {
            best_score = mean_score;
            best_params = config;
        }
    }
}

// Train final model with best parameters
SVMClassifier final_svm(best_params);
final_svm.fit(X_train, y_train);
```

#### 2. Model Evaluation
```cpp
SVMClassifier svm(config);
svm.fit(X_train, y_train);

// Get comprehensive metrics
auto metrics = svm.evaluate(X_test, y_test);

std::cout << "Accuracy: " << metrics.accuracy << std::endl;
std::cout << "Precision: " << metrics.precision << std::endl;
std::cout << "Recall: " << metrics.recall << std::endl;
std::cout << "F1-Score: " << metrics.f1_score << std::endl;

// Print confusion matrix
for (const auto& row : metrics.confusion_matrix) {
    for (int val : row) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```

#### 3. Probability Predictions
```cpp
json config = {
    {"kernel", "rbf"},
    {"C", 1.0},
    {"probability", true}  // Enable probability estimates
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);

// Get probability matrix (n_samples × n_classes)
auto proba = svm.predict_proba(X_test);

// Get decision values
auto decision = svm.decision_function(X_test);
```

#### 4. Feature Importance (Linear Kernels Only)
```cpp
json config = {{"kernel", "linear"}};
SVMClassifier svm(config);
svm.fit(X_train, y_train);

// Get feature weights
auto importance = svm.get_feature_importance();

// importance is a tensor of shape (n_features,)
// Higher absolute values indicate more important features
```

#### 5. Model Persistence
```cpp
// Train and save
SVMClassifier svm(config);
svm.fit(X_train, y_train);
svm.save_model("my_model.svm");

// Load and predict
SVMClassifier loaded_svm;
loaded_svm.load_model("my_model.svm");
auto predictions = loaded_svm.predict(X_test);
```

### Multiclass Examples

#### 1. One-vs-Rest Strategy
```cpp
// OvR is better for many classes and faster
json config = {
    {"kernel", "rbf"},
    {"C", 1.0},
    {"multiclass_strategy", "ovr"}
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);  // y_train can have multiple classes
```

#### 2. One-vs-One Strategy
```cpp
// OvO is better for imbalanced datasets
json config = {
    {"kernel", "rbf"},
    {"C", 10.0},
    {"multiclass_strategy", "ovo"}
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);
```

---

## Best Practices

### 1. Feature Preprocessing

**Always preprocess features before training:**

```cpp
// Standardization (zero mean, unit variance)
torch::Tensor standardize(const torch::Tensor& X) {
    auto mean = X.mean(0);
    auto std = X.std(0);
    std = torch::where(std == 0.0, torch::ones_like(std), std);
    return (X - mean) / std;
}

// Normalization ([0, 1] range)
torch::Tensor normalize(const torch::Tensor& X) {
    auto min_vals = std::get<0>(torch::min(X, 0));
    auto max_vals = std::get<0>(torch::max(X, 0));
    auto range = max_vals - min_vals;
    range = torch::where(range == 0.0, torch::ones_like(range), range);
    return (X - min_vals) / range;
}

// Apply before training
X_train = standardize(X_train);
X_test = standardize(X_test);
```

**Why?**
- RBF and polynomial kernels are sensitive to feature scales
- Prevents features with large magnitudes from dominating
- Improves numerical stability
- Speeds up convergence

### 2. Kernel Selection

**Decision Tree:**
```
Is the data linearly separable?
├─ Yes → Use LINEAR kernel
│         - Fastest training and prediction
│         - Works well for high-dimensional data (e.g., text)
│         - Provides feature importance
│
└─ No → Is the data sparse?
    ├─ Yes → Try LINEAR first, then RBF
    │
    └─ No → Use RBF kernel
              - Best general-purpose non-linear kernel
              - Good starting point
              - Tune C and gamma together

    Special cases:
    - Natural polynomial relationships → POLYNOMIAL
    - Want neural network-like behavior → SIGMOID
```

### 3. Hyperparameter Tuning Strategy

**Recommended approach:**

1. **Start simple:**
   ```cpp
   json config = {
       {"kernel", "linear"},
       {"C", 1.0}
   };
   ```

2. **If linear doesn't work well, try RBF with defaults:**
   ```cpp
   json config = {
       {"kernel", "rbf"},
       {"C", 1.0},
       {"gamma", "auto"}
   };
   ```

3. **Grid search over C and gamma:**
   ```cpp
   std::vector<double> c_values = {0.01, 0.1, 1.0, 10.0, 100.0};
   std::vector<double> gamma_values = {0.001, 0.01, 0.1, 1.0};
   // Try all combinations with cross-validation
   ```

4. **Fine-tune best region:**
   ```cpp
   // If best was C=10, gamma=0.1, search around it
   std::vector<double> c_values = {5.0, 10.0, 20.0, 50.0};
   std::vector<double> gamma_values = {0.05, 0.1, 0.2, 0.5};
   ```

### 4. Handling Class Imbalance

**Strategies:**

1. **Increase C parameter:**
   ```cpp
   json config = {
       {"kernel", "rbf"},
       {"C", 100.0},  // Higher C penalizes misclassification more
       {"multiclass_strategy", "ovo"}  // OvO handles imbalance better
   };
   ```

2. **Use appropriate multiclass strategy:**
   - Prefer OvO for imbalanced datasets
   - Monitor per-class metrics, not just overall accuracy

3. **Preprocessing:**
   - Consider resampling (oversample minority, undersample majority)
   - Use stratified cross-validation

### 5. Performance Optimization

**For large datasets:**

```cpp
json config = {
    {"kernel", "linear"},  // Faster than non-linear
    {"tolerance", 1e-2},   // Less strict convergence
    {"cache_size", 1000.0}, // Larger cache for RBF
    {"max_iterations", 1000}  // Limit iterations
};
```

**Memory considerations:**
- Linear kernel: Memory ~ O(n_features)
- RBF kernel: Memory ~ O(n_samples²) for kernel matrix
- Use cache_size to control memory for non-linear kernels

### 6. Cross-Validation

**Always use cross-validation for:**
- Hyperparameter selection
- Model comparison
- Performance estimation

```cpp
SVMClassifier svm(config);
auto cv_scores = svm.cross_validate(X, y, 5);  // 5-fold CV

double mean_score = std::accumulate(cv_scores.begin(),
                                   cv_scores.end(), 0.0) / cv_scores.size();
double std_score = /* calculate standard deviation */;

std::cout << "CV Score: " << mean_score << " ± " << std_score << std::endl;
```

### 7. Common Pitfalls to Avoid

❌ **Don't:**
- Forget to scale features before using RBF/polynomial kernels
- Use very high polynomial degrees (>5)
- Set C too high on noisy data (causes overfitting)
- Use OvO with many classes (N×(N-1)/2 classifiers is expensive)
- Ignore class imbalance

✅ **Do:**
- Always split data into train/validation/test sets
- Use cross-validation for hyperparameter tuning
- Monitor both training and test performance
- Start with simple models (linear) before complex ones
- Preprocess features consistently for train and test data

### 8. Parameter Relationships

**Critical interactions:**

1. **C and gamma (RBF kernel):**
   - High C + High gamma = Extreme overfitting
   - Low C + Low gamma = Extreme underfitting
   - Tune together, not independently

2. **Kernel complexity and C:**
   - More complex kernel (polynomial degree) → Lower C
   - Simpler kernel → Can use higher C

3. **Dataset size and cache_size:**
   - Small dataset (<10k samples) → cache_size = 200 MB sufficient
   - Large dataset (>100k samples) → increase cache_size to 500-1000 MB

---

## Quick Reference Table

| Parameter | Type | Default | Range | Kernels | Description |
|-----------|------|---------|-------|---------|-------------|
| `kernel` | string | "linear" | linear, rbf, polynomial, sigmoid | All | Kernel type |
| `C` | double | 1.0 | >0 | All | Regularization strength |
| `multiclass_strategy` | string | "ovr" | ovr, ovo | All | Multiclass approach |
| `gamma` | double/string | "auto" | >0 or "auto" | RBF, Poly, Sigmoid | Kernel coefficient |
| `degree` | int | 3 | ≥1 | Polynomial | Polynomial degree |
| `coef0` | double | 0.0 | any | Poly, Sigmoid | Independent term |
| `tolerance` | double | 1e-3 | >0 | All | Convergence tolerance |
| `max_iterations` | int | -1 | >0 or -1 | All | Max optimization iterations |
| `cache_size` | double | 200.0 | ≥0 | Non-linear | Kernel cache size (MB) |
| `probability` | bool | false | true/false | All | Enable probability estimates |

---

## Conclusion

This guide covers the complete functionality of SVMClassifier. For more examples, see:
- `examples/basic_usage.cpp` - Basic usage patterns
- `examples/advanced_usage.cpp` - Advanced techniques and hyperparameter tuning
- Unit tests in `tests/` directory - Comprehensive test cases

For issues or questions, refer to the project documentation or source code comments.
