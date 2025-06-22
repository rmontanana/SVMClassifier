# Development Guide

This guide provides comprehensive information for developers who want to contribute to the SVM Classifier C++ project.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Building from Source](#building-from-source)
- [Testing](#testing)
- [Code Style and Standards](#code-style-and-standards)
- [Contributing Guidelines](#contributing-guidelines)
- [Debugging and Profiling](#debugging-and-profiling)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Development Environment Setup

### Prerequisites

**Required:**
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- Git
- libtorch (PyTorch C++)
- pkg-config

**Optional (but recommended):**
- Doxygen (for documentation)
- Valgrind (for memory checking)
- lcov/gcov (for coverage analysis)
- clang-format (for code formatting)
- clang-tidy (for static analysis)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/svm-classifier.git
cd svm-classifier

# Run the automated setup
chmod +x install.sh
./install.sh --build-type Debug

# Or use the validation script for comprehensive testing
chmod +x validate_build.sh
./validate_build.sh --verbose --performance --memory-check
```

### Docker Development Environment

```bash
# Build development image
docker build --target development -t svm-dev .

# Run development container
docker run --rm -it -v $(pwd):/workspace svm-dev

# Inside container:
cd /workspace
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

## Project Structure

```
svm-classifier/
├── include/svm_classifier/     # Public header files
│   ├── svm_classifier.hpp      # Main classifier interface
│   ├── data_converter.hpp      # Tensor conversion utilities
│   ├── multiclass_strategy.hpp # Multiclass strategies
│   ├── kernel_parameters.hpp   # Parameter management
│   └── types.hpp               # Common types and enums
├── src/                        # Implementation files
│   ├── svm_classifier.cpp
│   ├── data_converter.cpp
│   ├── multiclass_strategy.cpp
│   └── kernel_parameters.cpp
├── tests/                      # Test suite
│   ├── test_main.cpp           # Test runner
│   ├── test_svm_classifier.cpp # Integration tests
│   ├── test_data_converter.cpp # Unit tests
│   ├── test_multiclass_strategy.cpp
│   ├── test_kernel_parameters.cpp
│   └── test_performance.cpp    # Performance benchmarks
├── examples/                   # Usage examples
│   ├── basic_usage.cpp
│   └── advanced_usage.cpp
├── external/                   # Third-party dependencies
├── cmake/                      # CMake configuration files
├── .github/workflows/          # CI/CD configuration
├── docs/                       # Documentation (generated)
├── CMakeLists.txt              # Main build configuration
├── Doxyfile                    # Documentation configuration
├── Dockerfile                  # Container configuration
├── README.md                   # Project overview
├── QUICK_START.md              # Getting started guide
├── DEVELOPMENT.md              # This file
└── LICENSE                     # License information
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SVMClassifier                            │
│  ┌─────────────────┐  ┌──────────────────────────────────┐  │
│  │ KernelParameters│  │      MulticlassStrategy          │  │
│  │                 │  │  ┌─────────────┐┌─────────────┐   │  │
│  │ - JSON config   │  │  │OneVsRest    ││OneVsOne     │   │  │
│  │ - Validation    │  │  │Strategy     ││Strategy     │   │  │
│  │ - Defaults      │  │  └─────────────┘└─────────────┘   │  │
│  └─────────────────┘  └──────────────────────────────────┘  │
│           │                         │                       │
│           └─────────┬───────────────┘                       │
│                     │                                       │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              DataConverter                              │  │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │  │
│  │  │ Tensor → libsvm │  │ Tensor → liblinear          │   │  │
│  │  │ Tensor → liblinear│ Results → Tensor            │   │  │
│  │  └─────────────────┘  └─────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │                              │
    ┌─────────────┐              ┌─────────────┐
    │   libsvm    │              │  liblinear  │
    │             │              │             │
    │ - RBF       │              │ - Linear    │
    │ - Polynomial│              │ - Fast      │
    │ - Sigmoid   │              │ - Scalable  │
    └─────────────┘              └─────────────┘
```

## Building from Source

### Debug Build

```bash
mkdir build-debug && cd build-debug
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_CXX_FLAGS="-g -O0 -Wall -Wextra"
make -j$(nproc)
```

### Release Build

```bash
mkdir build-release && cd build-release
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG"
make -j$(nproc)
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `CMAKE_BUILD_TYPE` | Build configuration | `Release` |
| `CMAKE_PREFIX_PATH` | PyTorch installation path | Auto-detect |
| `CMAKE_INSTALL_PREFIX` | Installation directory | `/usr/local` |
| `BUILD_TESTING` | Enable testing | `ON` |
| `BUILD_EXAMPLES` | Build examples | `ON` |

### Cross-Platform Building

#### Windows (MSVC)

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
  -DCMAKE_PREFIX_PATH=C:\libtorch ^
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build . --config Release
```

#### macOS

```bash
# Install dependencies with Homebrew
brew install cmake pkg-config openblas

# Build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch
make -j$(sysctl -n hw.ncpu)
```

## Testing

### Test Categories

- **Unit Tests** (`[unit]`): Test individual components
- **Integration Tests** (`[integration]`): Test component interactions
- **Performance Tests** (`[performance]`): Benchmark performance

### Running Tests

```bash
cd build

# Run all tests
ctest --output-on-failure

# Run specific test categories
./svm_classifier_tests "[unit]"
./svm_classifier_tests "[integration]"
./svm_classifier_tests "[performance]"

# Run with verbose output
./svm_classifier_tests "[unit]" --reporter console

# Run specific test
./svm_classifier_tests "SVMClassifier Construction"
```

### Coverage Analysis

```bash
# Build with coverage
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
make -j$(nproc)

# Run tests
./svm_classifier_tests

# Generate coverage report
make coverage

# View HTML report
open coverage_html/index.html
```

### Memory Testing

```bash
# Run with Valgrind
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
  ./svm_classifier_tests "[unit]"

# Or use the provided target
make test_memcheck
```

### Adding New Tests

1. **Unit Tests**: Add to appropriate `test_*.cpp` file
2. **Integration Tests**: Add to `test_svm_classifier.cpp`
3. **Performance Tests**: Add to `test_performance.cpp`

Example test structure:

```cpp
TEST_CASE("Feature Description", "[category][subcategory]") {
    SECTION("Specific behavior") {
        // Arrange
        auto svm = SVMClassifier(KernelType::LINEAR);
        auto X = torch::randn({100, 10});
        auto y = torch::randint(0, 2, {100});
        
        // Act
        auto metrics = svm.fit(X, y);
        
        // Assert
        REQUIRE(svm.is_fitted());
        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
    }
}
```

## Code Style and Standards

### C++ Standards

- **Language Standard**: C++17
- **Naming Convention**: `snake_case` for functions/variables, `PascalCase` for classes
- **File Naming**: `snake_case.hpp` and `snake_case.cpp`
- **Indentation**: 4 spaces (no tabs)

### Code Formatting

Use clang-format with the provided configuration:

```bash
# Format all source files
find src include tests examples -name "*.cpp" -o -name "*.hpp" | \
  xargs clang-format -i

# Check formatting
find src include tests examples -name "*.cpp" -o -name "*.hpp" | \
  xargs clang-format --dry-run --Werror
```

### Static Analysis

```bash
# Run clang-tidy
clang-tidy src/*.cpp include/svm_classifier/*.hpp \
  -- -I include -I /opt/libtorch/include
```

### Documentation Standards

- Use Doxygen-style comments for public APIs
- Include `@brief`, `@param`, `@return`, `@throws` as appropriate
- Provide usage examples for complex functions

Example:

```cpp
/**
 * @brief Train the SVM classifier on the provided dataset
 * @param X Feature tensor of shape (n_samples, n_features)
 * @param y Target tensor of shape (n_samples,) with class labels
 * @return Training metrics including timing and convergence info
 * @throws std::invalid_argument if input data is invalid
 * @throws std::runtime_error if training fails
 * 
 * @code
 * auto X = torch::randn({100, 4});
 * auto y = torch::randint(0, 3, {100});
 * SVMClassifier svm(KernelType::RBF, 1.0);
 * auto metrics = svm.fit(X, y);
 * @endcode
 */
TrainingMetrics fit(const torch::Tensor& X, const torch::Tensor& y);
```

### Error Handling

- Use exceptions for error conditions
- Provide meaningful error messages
- Validate inputs at public API boundaries
- Use RAII for resource management

### Performance Guidelines

- Minimize memory allocations in hot paths
- Use move semantics where appropriate
- Prefer algorithms from STL
- Profile before optimizing

## Contributing Guidelines

### Workflow

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Implement** your changes
4. **Add** tests for new functionality
5. **Run** the validation script: `./validate_build.sh`
6. **Commit** with descriptive messages
7. **Push** to your fork
8. **Create** a Pull Request

### Commit Message Format

```
type(scope): short description

Longer description if needed

- Bullet points for details
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`

### Pull Request Requirements

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Breaking changes are documented

### Code Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer review required
3. Address all review comments
4. Ensure branch is up-to-date with main

## Debugging and Profiling

### Debugging Builds

```bash
# Debug build with symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g -O0"

# Run with GDB
gdb ./svm_classifier_tests
(gdb) run "[unit]"
```

### Common Debugging Scenarios

```cpp
// Enable verbose logging (if implemented)
torch::set_num_threads(1);  // Single-threaded for reproducibility

// Print tensor information
std::cout << "X shape: " << X.sizes() << std::endl;
std::cout << "X dtype: " << X.dtype() << std::endl;
std::cout << "X device: " << X.device() << std::endl;

// Check for NaN/Inf values
if (torch::any(torch::isnan(X)).item<bool>()) {
    throw std::runtime_error("X contains NaN values");
}
```

### Performance Profiling

```bash
# Build with profiling
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Profile with perf
perf record ./svm_classifier_tests "[performance]"
perf report

# Profile with gprof
g++ -pg -o program program.cpp
./program
gprof program gmon.out > analysis.txt
```

## Documentation

### Building Documentation

```bash
# Generate API documentation
doxygen Doxyfile

# View documentation
open docs/html/index.html
```

### Documentation Structure

- **README.md**: Project overview and quick start
- **QUICK_START.md**: Step-by-step getting started guide
- **DEVELOPMENT.md**: This development guide
- **API Reference**: Generated from source code comments

### Contributing to Documentation

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include practical examples
- Test all code examples

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH**
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible new features
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. **Update version** in CMakeLists.txt
2. **Update CHANGELOG.md** with new features/fixes
3. **Run full validation**: `./validate_build.sh --performance --memory-check`
4. **Update documentation** if needed
5. **Create release tag**: `git tag -a v1.0.0 -m "Release 1.0.0"`
6. **Push tag**: `git push origin v1.0.0`
7. **Create GitHub release** with release notes
8. **Update package managers** (if applicable)

### Continuous Integration

Our CI/CD pipeline runs on every PR and includes:

- **Build testing** on multiple platforms (Ubuntu, macOS, Windows)
- **Compiler compatibility** (GCC, Clang, MSVC)
- **Code quality** checks (formatting, static analysis)
- **Test execution** (unit, integration, performance)
- **Coverage analysis**
- **Memory leak detection**
- **Documentation generation**
- **Package creation**

### Branch Strategy

- **main**: Stable releases
- **develop**: Integration branch for features
- **feature/***: Individual feature development
- **hotfix/***: Critical bug fixes
- **release/***: Release preparation

## Getting Help

### Resources

- 📖 [Project Documentation](README.md)
- 🐛 [Issue Tracker](https://github.com/your-username/svm-classifier/issues)
- 💬 [Discussions](https://github.com/your-username/svm-classifier/discussions)
- 📧 Email: svm-classifier@example.com

### Reporting Issues

When reporting issues, please include:

1. **Environment**: OS, compiler, library versions
2. **Reproduction**: Minimal code example
3. **Expected vs Actual**: What should happen vs what happens
4. **Logs**: Error messages, stack traces
5. **Investigation**: What you've tried already

### Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Propose an API** if applicable
4. **Consider implementation** complexity
5. **Offer to contribute** if possible

---

**Thank you for contributing to SVM Classifier C++! 🎯**