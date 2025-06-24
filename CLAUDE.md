# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System & Commands

This project uses CMake as the build system. The following commands are commonly used:

### Essential Build Commands
```bash
# Basic build (from project root)
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Build specific targets
make svm_classifier          # Main library
make svm_classifier_tests    # Test executable
make examples               # Example programs
```

### Automated Build Scripts
- `./install.sh` - Complete installation script with dependency management
- `./validate_build.sh` - Comprehensive build validation and testing
- `./build_docs.sh` - Generate Doxygen documentation

### Testing Commands
```bash
# Run all tests
ctest

# Run specific test categories (using Catch2 tags)
./svm_classifier_tests "[unit]"
./svm_classifier_tests "[integration]" 
./svm_classifier_tests "[performance]"

# Memory checks (if valgrind available)
valgrind --tool=memcheck --leak-check=full ./svm_classifier_tests "[unit]"
```

### Documentation Generation
```bash
# Generate docs (requires Doxygen)
cmake --build . --target doxygen

# Or use the helper script
./build_docs.sh --open
```

## Architecture Overview

This is a high-performance SVM classifier library with a scikit-learn compatible API that automatically selects optimal underlying implementations.

### Core Architecture
- **SVMClassifier**: Main classifier interface (`src/svm_classifier.cpp`)
- **DataConverter**: PyTorch tensor conversion utilities (`src/data_converter.cpp`)
- **MulticlassStrategy**: One-vs-Rest and One-vs-One implementations (`src/multiclass_strategy.cpp`)  
- **KernelParameters**: Parameter management and validation (`src/kernel_parameters.cpp`)

### Library Selection Logic
The classifier automatically chooses the best underlying implementation:
- **Linear kernels** → liblinear (highly optimized for linear classification)
- **Non-linear kernels** (RBF, Polynomial, Sigmoid) → libsvm (supports arbitrary kernels)

### Key Dependencies
- **libtorch**: PyTorch C++ API for tensor operations and integration
- **libsvm**: Non-linear SVM implementation (fetched via CMake)
- **liblinear**: Linear SVM implementation (fetched via CMake)
- **nlohmann/json**: JSON configuration handling (fetched via CMake)
- **Catch2**: Testing framework (fetched via CMake)

### Directory Structure
```
include/svm_classifier/     # Public API headers
src/                        # Implementation files
tests/                      # Comprehensive test suite using Catch2
examples/                   # Usage examples (basic_usage.cpp, advanced_usage.cpp)
external/                   # Build scripts for external dependencies
docs/                       # Doxygen configuration and output
```

## Development Workflow

### Adding New Features
1. Add implementation in `src/` directory
2. Add corresponding header in `include/svm_classifier/`
3. Write comprehensive tests in `tests/` using Catch2 framework
4. Add usage example in `examples/` if applicable
5. Update documentation with proper Doxygen comments

### Testing Strategy
The project uses Catch2 with comprehensive test coverage:
- Unit tests for individual components
- Integration tests for complete workflows  
- Performance benchmarks for optimization validation
- Memory leak detection with valgrind

### Code Style & Patterns
- Modern C++17 features and idioms
- RAII for resource management
- Exception-based error handling
- Comprehensive const-correctness
- Template-based generic programming where appropriate

## Configuration & Parameters

### JSON Configuration Support
The classifier accepts configuration via nlohmann::json:
```cpp
nlohmann::json config = {
    {"kernel", "rbf"},
    {"C", 1.0},
    {"gamma", "scale"},
    {"multiclass_strategy", "ovo"},
    {"probability", true}
};
SVMClassifier svm(config);
```

### PyTorch Integration
Native support for libtorch tensors:
- Input data as `torch::Tensor`
- Automatic data type conversion and validation
- GPU support through PyTorch's CUDA integration
- Memory-efficient tensor operations

### Multiclass Strategies
- **One-vs-Rest (OvR)**: Train N binary classifiers for N classes
- **One-vs-One (OvO)**: Train N*(N-1)/2 binary classifiers

## Performance Considerations

### Memory Management
- Automatic memory management for SVM structures
- Efficient sparse data handling
- Configurable cache sizes for large datasets
- RAII-based resource cleanup

### Optimization Features
- Automatic library selection for optimal performance
- Multi-threading support via libtorch
- Vectorized operations for data conversion
- Memory-mapped file support for large datasets

## Common Development Tasks

### Adding a New Kernel
1. Extend `KernelType` enum in `include/svm_classifier/types.hpp`
2. Add parameter validation in `kernel_parameters.cpp`
3. Update conversion logic in `data_converter.cpp`
4. Add comprehensive tests for the new kernel
5. Update examples and documentation

### Debugging Build Issues
- Use `./validate_build.sh --verbose` for detailed build validation
- Check PyTorch installation and `CMAKE_PREFIX_PATH`
- Verify all external dependencies are properly fetched
- Examine `build/CMakeFiles/CMakeError.log` for configuration issues

### Performance Profiling
```bash
# Build with profiling
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Profile with perf (Linux)
perf record --call-graph=dwarf ./svm_classifier_tests "[performance]"
perf report

# Memory profiling with valgrind
./validate_build.sh --memory-check
```