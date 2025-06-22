# Contributing to SVM Classifier C++

Thank you for your interest in contributing to SVM Classifier C++! This document provides guidelines and information for contributors.

## 🚀 Quick Start for Contributors

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/svm-classifier.git`
3. **Create** a branch: `git checkout -b feature/amazing-feature`
4. **Set up** development environment: `./install.sh --build-type Debug`
5. **Make** your changes
6. **Test** your changes: `./validate_build.sh`
7. **Commit** and **push**: `git commit -m "Add amazing feature" && git push origin feature/amazing-feature`
8. **Create** a Pull Request

## 🎯 Ways to Contribute

### 🐛 Bug Reports

Found a bug? Help us fix it!

- **Search existing issues** first to avoid duplicates
- **Use the bug report template** when creating new issues
- **Provide minimal reproduction** code when possible
- **Include system information** (OS, compiler, library versions)

### ✨ Feature Requests

Have an idea for improvement?

- **Check the roadmap** in issues to see if it's already planned
- **Use the feature request template**
- **Explain the use case** and why it would be valuable
- **Consider offering to implement** the feature yourself

### 📚 Documentation

Documentation improvements are always welcome!

- **Fix typos and grammar**
- **Add examples** for complex features
- **Improve API documentation** in source code
- **Write tutorials** for common use cases

### 🧪 Testing

Help us improve test coverage!

- **Add test cases** for edge cases
- **Improve performance tests**
- **Add integration tests** for real-world scenarios
- **Test on different platforms**

### 🔧 Code Contributions

Ready to dive into the code?

- **Follow our coding standards** (see below)
- **Add tests** for new functionality
- **Update documentation** as needed
- **Consider performance implications**

## 📋 Development Process

### Setting Up Development Environment

```bash
# 1. Clone and enter directory
git clone https://github.com/YOUR_USERNAME/svm-classifier.git
cd svm-classifier

# 2. Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake git pkg-config \
    libblas-dev liblapack-dev valgrind lcov doxygen

# 3. Set up development build
./install.sh --build-type Debug --verbose

# 4. Verify setup
./validate_build.sh --performance
```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/descriptive-name
   ```

2. **Make incremental commits**
   ```bash
   git add -A
   git commit -m "Add feature X: implement Y"
   ```

3. **Keep your branch updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Run tests frequently**
   ```bash
   cd build
   ./svm_classifier_tests
   ```

5. **Validate before pushing**
   ```bash
   ./validate_build.sh --memory-check
   ```

## 📏 Coding Standards

### Code Style

We use **clang-format** to ensure consistent formatting:

```bash
# Format your code before committing
find src include tests -name "*.cpp" -o -name "*.hpp" | \
    xargs clang-format -i

# Check formatting
find src include tests -name "*.cpp" -o -name "*.hpp" | \
    xargs clang-format --dry-run --Werror
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `SVMClassifier`, `DataConverter`)
- **Functions/Methods**: `snake_case` (e.g., `fit()`, `predict_proba()`)
- **Variables**: `snake_case` (e.g., `kernel_type_`, `n_features_`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TOLERANCE`)
- **Files**: `snake_case.hpp` and `snake_case.cpp`

### Code Organization

- **Header files**: Put in `include/svm_classifier/`
- **Implementation files**: Put in `src/`
- **Test files**: Put in `tests/` with `test_` prefix
- **Examples**: Put in `examples/`

### Documentation

All public APIs must be documented with Doxygen:

```cpp
/**
 * @brief Brief description of the function
 * @param param_name Description of parameter
 * @return Description of return value
 * @throws std::exception_type When this exception is thrown
 * 
 * Detailed description with usage example:
 * @code
 * SVMClassifier svm(KernelType::RBF);
 * auto metrics = svm.fit(X, y);
 * @endcode
 */
TrainingMetrics fit(const torch::Tensor& X, const torch::Tensor& y);
```

### Error Handling

- **Use exceptions** for error conditions
- **Provide meaningful messages** with context
- **Validate inputs** at public API boundaries
- **Use standard exception types** when appropriate

```cpp
if (X.size(0) == 0) {
    throw std::invalid_argument("X cannot have 0 samples");
}

if (!is_fitted_) {
    throw std::runtime_error("Model must be fitted before prediction");
}
```

### Performance Guidelines

- **Minimize allocations** in hot paths
- **Use move semantics** for expensive objects
- **Prefer STL algorithms** over manual loops
- **Profile before optimizing**
- **Consider memory usage** for large datasets

## 🧪 Testing Guidelines

### Test Categories

Mark your tests with appropriate tags:

- `[unit]`: Test individual components in isolation
- `[integration]`: Test component interactions
- `[performance]`: Benchmark performance characteristics

### Writing Good Tests

```cpp
TEST_CASE("Clear description of what is being tested", "[unit][component]") {
    SECTION("Specific behavior being verified") {
        // Arrange - Set up test data
        auto X = torch::randn({100, 4});
        auto y = torch::randint(0, 3, {100});
        SVMClassifier svm(KernelType::LINEAR);
        
        // Act - Perform the operation
        auto metrics = svm.fit(X, y);
        
        // Assert - Verify the results
        REQUIRE(svm.is_fitted());
        REQUIRE(metrics.status == TrainingStatus::SUCCESS);
        REQUIRE(metrics.training_time >= 0.0);
    }
}
```

### Test Requirements

- **All new public methods** must have tests
- **Edge cases** should be covered
- **Error conditions** should be tested
- **Performance regressions** should be prevented

### Running Tests

```bash
cd build

# Run all tests
./svm_classifier_tests

# Run specific categories
./svm_classifier_tests "[unit]"
./svm_classifier_tests "[integration]"

# Run with verbose output
./svm_classifier_tests --reporter console

# Generate coverage report
make coverage
```

## 📝 Commit Message Format

Use conventional commit format:

```
type(scope): brief description

Optional longer description explaining the change in more detail.

- Additional details as bullet points
- Reference issues: Fixes #123, Closes #456
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without functional changes
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

### Examples

```bash
git commit -m "feat(classifier): add polynomial kernel support

- Implement polynomial kernel with configurable degree
- Add comprehensive test coverage
- Update documentation with usage examples
- Fixes #42"

git commit -m "fix(converter): handle empty tensors gracefully

Previously, empty tensors would cause segmentation fault.
Now properly validates input and throws meaningful exception.

Fixes #89"

git commit -m "docs(readme): add installation troubleshooting section"

git commit -m "test(performance): add benchmarks for large datasets"
```

## 🔍 Pull Request Process

### Before Submitting

- [ ] **Rebase** on latest main branch
- [ ] **Run full validation**: `./validate_build.sh --performance --memory-check`
- [ ] **Update documentation** if needed
- [ ] **Add/update tests** for changes
- [ ] **Check code formatting**
- [ ] **Write descriptive commit messages**

### Pull Request Template

When creating a PR, please:

1. **Use a descriptive title** that summarizes the change
2. **Fill out the PR template** completely
3. **Link related issues** (e.g., "Fixes #123")
4. **Describe testing performed**
5. **Note any breaking changes**

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **At least one maintainer** review required
3. **Address all feedback** before merging
4. **Squash commits** if requested
5. **Update branch** if main has changed

### Review Criteria

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it well-written and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is documentation updated?
- **Performance**: No significant regressions?
- **Compatibility**: Works across supported platforms?

## 🏗️ Development Environment Setup

### Required Tools

- **C++17 compiler**: GCC 7+, Clang 5+, or MSVC 2019+
- **CMake**: 3.15 or later
- **Git**: For version control
- **PyTorch C++**: libtorch library

### Optional Tools

- **clang-format**: Code formatting
- **clang-tidy**: Static analysis
- **Valgrind**: Memory debugging
- **lcov**: Code coverage
- **Doxygen**: Documentation generation
- **Docker**: Containerized development

### IDE Configuration

#### Visual Studio Code

Recommended extensions:
- C/C++ (Microsoft)
- CMake Tools
- GitLens
- Doxygen Documentation Generator

#### CLion

Project configuration:
- CMake profile with Debug/Release configurations
- Code style settings for clang-format
- Valgrind integration for memory debugging

#### Visual Studio

Use CMake integration:
- Open folder with CMakeLists.txt
- Configure CMake settings for libtorch path
- Set up debugging configuration

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible new features  
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

- [ ] Update version in CMakeLists.txt
- [ ] Update CHANGELOG.md
- [ ] Run comprehensive validation
- [ ] Update documentation
- [ ] Create and test packages
- [ ] Create GitHub release
- [ ] Announce release

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please:

- **Be respectful** and inclusive
- **Be patient** with newcomers
- **Be constructive** in feedback
- **Focus on what is best** for the community
- **Show empathy** towards other members

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code review and discussion
- **Email**: Direct contact for sensitive issues

### Getting Help

Stuck? Here's how to get help:

1. **Check the documentation**: README, QUICK_START, DEVELOPMENT guides
2. **Search existing issues**: Your question might already be answered
3. **Ask in discussions**: For general questions and advice
4. **Create an issue**: For specific bugs or feature requests

## 📞 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-username/svm-classifier/issues)
- **GitHub Discussions**: [Community discussions](https://github.com/your-username/svm-classifier/discussions)
- **Email**: svm-classifier@example.com

## 🙏 Recognition

Contributors are recognized in:

- **CHANGELOG.md**: Major contributions listed in releases
- **README.md**: Contributors section
- **GitHub**: Contributor statistics and graphs

Thank you for contributing to SVM Classifier C++! Your efforts help make this project better for everyone. 🎯