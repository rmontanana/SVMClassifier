# SVM Classifier C++

A high-performance Support Vector Machine classifier implementation in C++ with a scikit-learn compatible API. This library provides a unified interface for SVM classification using both liblinear (for linear kernels) and libsvm (for non-linear kernels), with support for multiclass classification and PyTorch tensor integration.

## Features

- **🚀 Scikit-learn Compatible API**: Familiar `fit()`, `predict()`, `predict_proba()`, `score()` methods
- **🔧 Multiple Kernels**: Linear, RBF, Polynomial, and Sigmoid kernels
- **📊 Multiclass Support**: One-vs-Rest (OvR) and One-vs-One (OvO) strategies
- **⚡ Automatic Library Selection**: Uses liblinear for linear kernels, libsvm for others
- **🔗 PyTorch Integration**: Native support for libtorch tensors
- **⚙️ JSON Configuration**: Easy parameter management with nlohmann::json
- **🧪 Comprehensive Testing**: 100% test coverage with Catch2
- **📈 Performance Metrics**: Detailed evaluation and training metrics
- **🔍 Cross-Validation**: Built-in k-fold cross-validation support
- **🎯 Grid Search**: Hyperparameter optimization capabilities

## Quick Start

### Prerequisites

- C++17 or later
- CMake 3.15+
- libtorch
- Git

### Building

```bash
git clone <repository-url>
cd svm_classifier
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Basic Usage

```cpp
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>

using namespace svm_classifier;

// Create sample data
auto X = torch::randn({100, 2});  // 100 samples, 2 features
auto y = torch::randint(0, 3, {100}); // 3 classes

// Create and train SVM
SVMClassifier svm(KernelType::RBF, 1.0);
auto metrics = svm.fit(X, y);

// Make predictions
auto predictions = svm.predict(X);
auto probabilities = svm.predict_proba(X);
double accuracy = svm.score(X, y);
```

### JSON Configuration

```cpp
#include <nlohmann/json.hpp>

nlohmann::json config = {
    {"kernel", "rbf"},
    {"C", 10.0},
    {"gamma", 0.1},
    {"multiclass_strategy", "ovo"},
    {"probability", true}
};

SVMClassifier svm(config);
```

## API Reference

### Constructor Options

```cpp
// Default constructor
SVMClassifier svm;

// With explicit parameters
SVMClassifier svm(KernelType::RBF, 1.0, MulticlassStrategy::ONE_VS_REST);

// From JSON configuration
SVMClassifier svm(config_json);
```

### Core Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(X, y)` | Train the classifier | `TrainingMetrics` |
| `predict(X)` | Predict class labels | `torch::Tensor` |
| `predict_proba(X)` | Predict class probabilities | `torch::Tensor` |
| `score(X, y)` | Calculate accuracy | `double` |
| `decision_function(X)` | Get decision values | `torch::Tensor` |
| `cross_validate(X, y, cv)` | K-fold cross-validation | `std::vector<double>` |
| `grid_search(X, y, grid, cv)` | Hyperparameter tuning | `nlohmann::json` |

### Parameter Configuration

#### Common Parameters
- **kernel**: `"linear"`, `"rbf"`, `"polynomial"`, `"sigmoid"`
- **C**: Regularization parameter (default: 1.0)
- **multiclass_strategy**: `"ovr"` (One-vs-Rest) or `"ovo"` (One-vs-One)
- **probability**: Enable probability estimates (default: false)
- **tolerance**: Convergence tolerance (default: 1e-3)

#### Kernel-Specific Parameters
- **RBF/Polynomial/Sigmoid**: `gamma` (default: auto)
- **Polynomial**: `degree` (default: 3), `coef0` (default: 0.0)
- **Sigmoid**: `coef0` (default: 0.0)

## Examples

### Multi-class Classification

```cpp
// Generate multi-class dataset
auto X = torch::randn({300, 4});
auto y = torch::randint(0, 5, {300});  // 5 classes

// Configure for multi-class
nlohmann::json config = {
    {"kernel", "rbf"},
    {"C", 1.0},
    {"gamma", 0.1},
    {"multiclass_strategy", "ovo"},
    {"probability", true}
};

SVMClassifier svm(config);
auto metrics = svm.fit(X, y);

// Evaluate
auto eval_metrics = svm.evaluate(X, y);
std::cout << "Accuracy: " << eval_metrics.accuracy << std::endl;
std::cout << "F1-Score: " << eval_metrics.f1_score << std::endl;
```

### Cross-Validation

```cpp
SVMClassifier svm(KernelType::RBF);
auto cv_scores = svm.cross_validate(X, y, 5);  // 5-fold CV

double mean_score = 0.0;
for (auto score : cv_scores) {
    mean_score += score;
}
mean_score /= cv_scores.size();
```

### Grid Search

```cpp
nlohmann::json param_grid = {
    {"C", {0.1, 1.0, 10.0}},
    {"gamma", {0.01, 0.1, 1.0}},
    {"kernel", {"rbf", "polynomial"}}
};

auto best_params = svm.grid_search(X, y, param_grid, 3);
std::cout << "Best parameters: " << best_params.dump(2) << std::endl;
```

## Testing

### Run All Tests

```bash
cd build
make test_all
```

### Test Categories

```bash
make test_unit          # Unit tests
make test_integration   # Integration tests  
make test_performance   # Performance tests
```

### Coverage Report

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make coverage
```

The coverage report will be generated in `build/coverage_html/index.html`.

## Project Structure

```
svm_classifier/
├── include/svm_classifier/     # Public headers
│   ├── svm_classifier.hpp      # Main classifier interface
│   ├── data_converter.hpp      # Tensor conversion utilities
│   ├── multiclass_strategy.hpp # Multiclass strategies
│   ├── kernel_parameters.hpp   # Parameter management
│   └── types.hpp               # Common types and enums
├── src/                        # Implementation files
├── tests/                      # Comprehensive test suite
├── examples/                   # Usage examples
├── external/                   # Third-party dependencies
└── CMakeLists.txt             # Build configuration
```

## Dependencies

### Required
- **libtorch**: PyTorch C++ API for tensor operations
- **liblinear**: Linear SVM implementation
- **libsvm**: Non-linear SVM implementation
- **nlohmann/json**: JSON configuration handling

### Testing
- **Catch2**: Testing framework

### Build System
- **CMake**: Cross-platform build system

## Performance Characteristics

### Memory Usage
- Efficient sparse data handling
- Automatic memory management for SVM structures
- Configurable cache sizes for large datasets

### Speed
- Linear kernels: Uses highly optimized liblinear
- Non-linear kernels: Uses proven libsvm implementation
- Multi-threading support via libtorch

### Scalability
- Handles datasets from hundreds to millions of samples
- Memory-efficient data conversion
- Sparse feature support

## Library Selection Logic

The classifier automatically selects the appropriate underlying library:

- **Linear Kernel** → liblinear (optimized for linear classification)
- **RBF/Polynomial/Sigmoid** → libsvm (supports arbitrary kernels)

This ensures optimal performance for each kernel type while maintaining a unified API.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `make test_all`
5. Check code coverage: `make coverage`
6. Submit a pull request

### Code Style

- Follow modern C++17 conventions
- Use RAII for resource management
- Comprehensive error handling
- Document all public APIs

## License

[Specify your license here]

## Acknowledgments

- **libsvm**: Chih-Chung Chang and Chih-Jen Lin
- **liblinear**: R.-E. Fan et al.
- **PyTorch**: Facebook AI Research
- **nlohmann/json**: Niels Lohmann
- **Catch2**: Phil Nash and contributors