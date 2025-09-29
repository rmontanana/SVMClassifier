# SVMClassifier Project Makefile
# Provides convenient targets for common development tasks

# Configuration
BUILD_DIR := build
BUILD_TYPE := Release
CMAKE_PREFIX_PATH ?= /opt/homebrew/lib/python3.11/site-packages/torch
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Detect PyTorch installation
TORCH_PATHS := \
	/opt/homebrew/lib/python3.11/site-packages/torch \
	/opt/homebrew/lib/python3.12/site-packages/torch \
	/usr/local/lib/python3.11/site-packages/torch \
	$(shell python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null) \
	./libtorch

define find_torch
$(foreach path,$(TORCH_PATHS),$(if $(wildcard $(path)/share/cmake/Torch),$(path),))
endef

DETECTED_TORCH := $(firstword $(call find_torch))
ifneq ($(DETECTED_TORCH),)
	CMAKE_PREFIX_PATH := $(DETECTED_TORCH)
endif

# Default target
.PHONY: all
all: help

# Help target
.PHONY: help
help:
	@echo "SVMClassifier Development Makefile"
	@echo ""
	@echo "Build Targets:"
	@echo "  build         - Build the project (Release)"
	@echo "  debug         - Build with debug symbols"
	@echo "  clean         - Clean build directory"
	@echo "  rebuild       - Clean and build"
	@echo "  install       - Run full installation with dependencies"
	@echo ""
	@echo "Testing Targets:"
	@echo "  test          - Run all tests (direct execution)"
	@echo "  test-ctest    - Run all tests (via CTest)"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests only"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  test-memcheck - Run tests with Valgrind memory checking"
	@echo "  test-profile  - Run tests with performance profiling"
	@echo ""
	@echo "Documentation Targets:"
	@echo "  docs          - Generate Doxygen documentation"
	@echo "  docs-open     - Generate and open documentation"
	@echo ""
	@echo "Development Targets:"
	@echo "  format        - Format code with clang-format"
	@echo "  lint          - Run static analysis with clang-tidy"
	@echo "  validate      - Run comprehensive build validation"
	@echo "  examples      - Build and run examples"
	@echo ""
	@echo "Utility Targets:"
	@echo "  info          - Show build configuration info"
	@echo "  deps          - Show project dependencies"
	@echo "  clean-all     - Clean everything including external deps"
	@echo ""
	@echo "Configuration:"
	@echo "  BUILD_TYPE=$(BUILD_TYPE)"
	@echo "  CMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH)"
	@echo "  JOBS=$(JOBS)"

# Build targets
.PHONY: build
build: $(BUILD_DIR)/Makefile
	@echo "Building SVMClassifier ($(BUILD_TYPE))..."
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)

.PHONY: debug
debug:
	@$(MAKE) build BUILD_TYPE=Debug BUILD_DIR=build_debug

$(BUILD_DIR)/Makefile:
	@echo "Configuring CMake build..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH) \
		-DBUILD_DOCUMENTATION=ON

.PHONY: clean
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

.PHONY: rebuild
rebuild: clean build

.PHONY: install
install:
	@echo "Running full installation..."
	@./install.sh

# Testing targets
.PHONY: test
test: build
	@echo "Running all tests..."
	@$(BUILD_DIR)/tests/svm_classifier_tests

.PHONY: test-ctest
test-ctest: build
	@echo "Running tests with CTest..."
	@cd $(BUILD_DIR) && ctest --output-on-failure --parallel $(JOBS)

.PHONY: test-unit
test-unit: build
	@echo "Running unit tests..."
	@$(BUILD_DIR)/tests/svm_classifier_tests "[unit]"

.PHONY: test-integration
test-integration: build
	@echo "Running integration tests..."
	@$(BUILD_DIR)/tests/svm_classifier_tests "[integration]"

.PHONY: test-performance
test-performance: build
	@echo "Running performance tests..."
	@$(BUILD_DIR)/tests/svm_classifier_tests "[performance]"

.PHONY: test-verbose
test-verbose: build
	@echo "Running tests with verbose output..."
	@$(BUILD_DIR)/tests/svm_classifier_tests -s

.PHONY: test-memcheck
test-memcheck: build
	@echo "Running memory check tests..."
	@cd $(BUILD_DIR) && $(MAKE) test_memcheck

.PHONY: test-profile
test-profile: build
	@echo "Running performance profiling..."
	@cd $(BUILD_DIR) && $(MAKE) test_profile

# Documentation targets
.PHONY: docs
docs: $(BUILD_DIR)/Makefile
	@echo "Generating documentation..."
	@cd $(BUILD_DIR) && $(MAKE) doxygen

.PHONY: docs-open
docs-open: docs
	@echo "Opening documentation..."
	@./build_docs.sh --open

# Development targets
.PHONY: format
format:
	@echo "Formatting code with clang-format..."
	@find src include tests examples -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

.PHONY: lint
lint:
	@echo "Running static analysis with clang-tidy..."
	@clang-tidy src/* -- -I include -std=c++17

.PHONY: validate
validate:
	@echo "Running comprehensive build validation..."
	@./validate_build.sh

.PHONY: examples
examples: build
	@echo "Building examples..."
	@cd $(BUILD_DIR) && $(MAKE) examples
	@echo "Running basic example..."
	@$(BUILD_DIR)/examples/basic_usage
	@echo "Running advanced example..."
	@$(BUILD_DIR)/examples/advanced_usage

# Utility targets
.PHONY: info
info:
	@echo "Build Configuration:"
	@echo "  Project: SVMClassifier"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Build Directory: $(BUILD_DIR)"
	@echo "  Jobs: $(JOBS)"
	@echo "  CMAKE_PREFIX_PATH: $(CMAKE_PREFIX_PATH)"
	@echo "  Detected PyTorch: $(DETECTED_TORCH)"
	@echo ""
	@echo "Available Tools:"
	@which cmake >/dev/null 2>&1 && echo "  ✓ CMake: $$(cmake --version | head -n1)" || echo "  ✗ CMake: Not found"
	@which doxygen >/dev/null 2>&1 && echo "  ✓ Doxygen: $$(doxygen --version)" || echo "  ✗ Doxygen: Not found"
	@which valgrind >/dev/null 2>&1 && echo "  ✓ Valgrind: $$(valgrind --version | head -n1)" || echo "  ✗ Valgrind: Not found"
	@which clang-format >/dev/null 2>&1 && echo "  ✓ clang-format: $$(clang-format --version | head -n1)" || echo "  ✗ clang-format: Not found"
	@which clang-tidy >/dev/null 2>&1 && echo "  ✓ clang-tidy: $$(clang-tidy --version | head -n1)" || echo "  ✗ clang-tidy: Not found"

.PHONY: deps
deps:
	@echo "Project Dependencies:"
	@echo "  Required:"
	@echo "    - CMake (>= 3.15)"
	@echo "    - C++17 compiler (GCC/Clang)"
	@echo "    - PyTorch/libtorch"
	@echo "  Fetched Automatically:"
	@echo "    - libsvm (v332)"
	@echo "    - liblinear (v249)"
	@echo "    - nlohmann/json (v3.11.3)"
	@echo "    - Catch2 (v3.4.0)"
	@echo "  Optional:"
	@echo "    - Doxygen (for documentation)"
	@echo "    - Valgrind (for memory checking)"
	@echo "    - clang-format (for code formatting)"
	@echo "    - clang-tidy (for static analysis)"

.PHONY: clean-all
clean-all:
	@echo "Cleaning all build directories and dependencies..."
	@rm -rf build build_Debug build_Release
	@rm -rf _deps
	@echo "All build artifacts cleaned."

# Aliases for convenience
.PHONY: configure
configure: $(BUILD_DIR)/Makefile

.PHONY: tests
tests: test

.PHONY: documentation
documentation: docs

.PHONY: check
check: test

.PHONY: memcheck
memcheck: test-memcheck

.PHONY: profile
profile: test-profile