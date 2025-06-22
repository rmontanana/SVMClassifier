# Quick Start Guide

Get up and running with SVM Classifier C++ in minutes!

## 🚀 One-Line Installation

```bash
curl -fsSL https://raw.githubusercontent.com/your-username/svm-classifier/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/your-username/svm-classifier.git
cd svm-classifier
chmod +x install.sh
./install.sh
```

## 📋 Prerequisites

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config libblas-dev liblapack-dev
```

### CentOS/RHEL
```bash
sudo yum install -y gcc-c++ cmake git pkgconfig blas-devel lapack-devel
```

### macOS
```bash
brew install cmake git pkg-config openblas
```

## 🔧 Manual Build

```bash
# 1. Clone the repository
git clone https://github.com/your-username/svm-classifier.git
cd svm-classifier

# 2. Install PyTorch C++ (if not already installed)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# 3. Build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=../libtorch
make -j$(nproc)

# 4. Run tests
make test

# 5. Install (optional)
sudo make install
```

## 💻 First Example

Create `my_svm.cpp`:

```cpp
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>
#include <iostream>

int main() {
    using namespace svm_classifier;
    
    // Create sample data
    auto X = torch::randn({100, 4});  // 100 samples, 4 features
    auto y = torch::randint(0, 3, {100});  // 3 classes
    
    // Create and train SVM
    SVMClassifier svm(KernelType::RBF, 1.0);
    auto metrics = svm.fit(X, y);
    
    // Make predictions
    auto predictions = svm.predict(X);
    double accuracy = svm.score(X, y);
    
    std::cout << "Training time: " << metrics.training_time << " seconds\n";
    std::cout << "Accuracy: " << (accuracy * 100) << "%\n";
    
    return 0;
}
```

Compile and run:

```bash
# If installed system-wide
g++ -std=c++17 my_svm.cpp -lsvm_classifier -ltorch -ltorch_cpu -o my_svm
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
./my_svm

# If built locally
g++ -std=c++17 -I../include -I../libtorch/include -I../libtorch/include/torch/csrc/api/include \
    my_svm.cpp -L../build -lsvm_classifier -L../libtorch/lib -ltorch -ltorch_cpu -o my_svm
./my_svm
```

## 🏗️ CMake Integration

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)

# Find packages
find_package(SVMClassifier REQUIRED)
find_package(Torch REQUIRED)

# Create executable
add_executable(my_svm my_svm.cpp)

# Link libraries
target_link_libraries(my_svm 
    SVMClassifier::svm_classifier
    ${TORCH_LIBRARIES}
)
```

Build:

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/usr/local;/opt/libtorch"
make
```

## 🎯 Common Use Cases

### Binary Classification

```cpp
#include <svm_classifier/svm_classifier.hpp>
#include <nlohmann/json.hpp>

// Configure for binary classification
nlohmann::json config = {
    {"kernel", "linear"},
    {"C", 1.0},
    {"probability", true}
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);

// Get predictions and probabilities
auto predictions = svm.predict(X_test);
auto probabilities = svm.predict_proba(X_test);
```

### Multiclass with RBF Kernel

```cpp
nlohmann::json config = {
    {"kernel", "rbf"},
    {"C", 10.0},
    {"gamma", 0.1},
    {"multiclass_strategy", "ovo"}  // One-vs-One
};

SVMClassifier svm(config);
svm.fit(X_train, y_train);

auto eval_metrics = svm.evaluate(X_test, y_test);
std::cout << "F1-Score: " << eval_metrics.f1_score << std::endl;
```

### Cross-Validation

```cpp
SVMClassifier svm(KernelType::RBF, 1.0);

// 5-fold cross-validation
auto cv_scores = svm.cross_validate(X, y, 5);

double mean_score = 0.0;
for (double score : cv_scores) {
    mean_score += score;
}
mean_score /= cv_scores.size();

std::cout << "CV Score: " << mean_score << " ± " << std_dev << std::endl;
```

### Hyperparameter Tuning

```cpp
nlohmann::json param_grid = {
    {"kernel", {"linear", "rbf"}},
    {"C", {0.1, 1.0, 10.0}},
    {"gamma", {0.01, 0.1, 1.0}}
};

auto results = svm.grid_search(X_train, y_train, param_grid, 3);
auto best_params = results["best_params"];

std::cout << "Best parameters: " << best_params.dump(2) << std::endl;
```

## 🐳 Docker Usage

```bash
# Build and run
docker build -t svm-classifier .
docker run --rm -it svm-classifier

# Development environment
docker build --target development -t svm-dev .
docker run --rm -it -v $(pwd):/workspace svm-dev
```

## 🧪 Running Tests

```bash
cd build

# All tests
make test_all

# Specific test categories
make test_unit          # Unit tests only
make test_integration   # Integration tests only
make test_performance   # Performance benchmarks

# With coverage (Debug build)
make coverage
```

## 📊 Performance Tips

1. **Kernel Selection**:
   - Linear: Fast, good for high-dimensional data
   - RBF: Good general-purpose choice
   - Polynomial: Good for non-linear patterns
   - Sigmoid: Similar to neural networks

2. **Multiclass Strategy**:
   - One-vs-Rest (OvR): Faster training, less memory
   - One-vs-One (OvO): Often better accuracy

3. **Data Preprocessing**:
   - Normalize features to [0,1] or standardize
   - Handle missing values
   - Consider feature selection

```cpp
// Example preprocessing
auto X_normalized = (X - X.mean(0)) / X.std(0);
```

## 🔧 Troubleshooting

### Common Issues

**Problem**: `undefined reference to torch::*`
**Solution**: Make sure libtorch is in your library path
```bash
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

**Problem**: CMake can't find SVMClassifier
**Solution**: Set the install prefix in CMAKE_PREFIX_PATH
```bash
cmake .. -DCMAKE_PREFIX_PATH="/usr/local;/opt/libtorch"
```

**Problem**: Compilation errors with C++17
**Solution**: Ensure your compiler supports C++17
```bash
g++ --version  # Should be 7.0+
clang++ --version  # Should be 5.0+
```

### Build Options

```bash
# Debug build with full debugging info
./install.sh --build-type Debug --verbose

# Custom installation directory
./install.sh --prefix ~/.local

# Skip tests for faster installation
./install.sh --skip-tests

# Clean build
./install.sh --clean
```

## 📚 Next Steps

- Check the [examples/](examples/) directory for more examples
- Read the [API documentation](docs/) for detailed reference
- Explore [advanced features](README.md#features) in the main README
- Join our [community discussions](https://github.com/your-username/svm-classifier/discussions)

## 🆘 Getting Help

- 📖 [Full Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/your-username/svm-classifier/issues)
- 💬 [Discussions](https://github.com/your-username/svm-classifier/discussions)
- 📧 [Contact](mailto:your-email@example.com)

---

**Happy classifying! 🎯**