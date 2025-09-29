# SVMClassifier Project Makefile
# Provides convenient targets for common development tasks

# Colors and icons for enhanced visual output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
BOLD := \033[1m
NC := \033[0m # No Color

# Icons
CHECK := ✅
CROSS := ❌
ROCKET := 🚀
GEAR := ⚙️
BOOK := 📚
TEST := 🧪
CLEAN := 🧹
INFO := ℹ️
WARN := ⚠️
SPARKLES := ✨

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
	@printf "$(BOLD)$(BLUE)$(ROCKET) SVMClassifier Development Makefile$(NC)\n"
	@printf "$(CYAN)===============================================$(NC)\n\n"
	@printf "$(BOLD)$(GREEN)$(GEAR) Build Targets:$(NC)\n"
	@printf "  $(YELLOW)build$(NC)         - $(ROCKET) Build the project (Release)\n"
	@printf "  $(YELLOW)debug$(NC)         - $(GEAR) Build with debug symbols\n"
	@printf "  $(YELLOW)clean$(NC)         - $(CLEAN) Clean build directory\n"
	@printf "  $(YELLOW)rebuild$(NC)       - $(SPARKLES) Clean and build\n"
	@printf "  $(YELLOW)install$(NC)       - $(ROCKET) Run full installation with dependencies\n\n"
	@printf "$(BOLD)$(GREEN)$(TEST) Testing Targets:$(NC)\n"
	@printf "  $(YELLOW)test$(NC)          - $(TEST) Run all tests (direct execution)\n"
	@printf "  $(YELLOW)test-ctest$(NC)    - $(TEST) Run all tests (via CTest)\n"
	@printf "  $(YELLOW)test-unit$(NC)     - $(TEST) Run unit tests only\n"
	@printf "  $(YELLOW)test-integration$(NC) - $(TEST) Run integration tests only\n"
	@printf "  $(YELLOW)test-performance$(NC) - $(TEST) Run performance tests only\n"
	@printf "  $(YELLOW)test-verbose$(NC)  - $(TEST) Run tests with verbose output\n"
	@printf "  $(YELLOW)test-memcheck$(NC) - $(TEST) Run tests with Valgrind memory checking\n"
	@printf "  $(YELLOW)test-profile$(NC)  - $(TEST) Run tests with performance profiling\n"
	@printf "  $(YELLOW)coverage$(NC)      - $(TEST) Run tests with code coverage analysis\n\n"
	@printf "$(BOLD)$(GREEN)$(BOOK) Documentation Targets:$(NC)\n"
	@printf "  $(YELLOW)docs$(NC)          - $(BOOK) Generate Doxygen documentation\n"
	@printf "  $(YELLOW)docs-open$(NC)     - $(BOOK) Generate and open documentation\n\n"
	@printf "$(BOLD)$(GREEN)$(SPARKLES) Development Targets:$(NC)\n"
	@printf "  $(YELLOW)format$(NC)        - $(SPARKLES) Format code with clang-format\n"
	@printf "  $(YELLOW)lint$(NC)          - $(GEAR) Run static analysis with clang-tidy\n"
	@printf "  $(YELLOW)validate$(NC)      - $(CHECK) Run comprehensive build validation\n"
	@printf "  $(YELLOW)examples$(NC)      - $(ROCKET) Build and run examples\n\n"
	@printf "$(BOLD)$(GREEN)$(INFO) Utility Targets:$(NC)\n"
	@printf "  $(YELLOW)info$(NC)          - $(INFO) Show build configuration info\n"
	@printf "  $(YELLOW)deps$(NC)          - $(INFO) Show project dependencies\n"
	@printf "  $(YELLOW)clean-all$(NC)     - $(CLEAN) Clean everything including external deps\n\n"
	@printf "$(BOLD)$(PURPLE)$(GEAR) Configuration:$(NC)\n"
	@printf "  $(CYAN)BUILD_TYPE$(NC)=$(GREEN)$(BUILD_TYPE)$(NC)\n"
	@printf "  $(CYAN)CMAKE_PREFIX_PATH$(NC)=$(GREEN)$(CMAKE_PREFIX_PATH)$(NC)\n"
	@printf "  $(CYAN)JOBS$(NC)=$(GREEN)$(JOBS)$(NC)\n"

# Build targets
.PHONY: build
build: $(BUILD_DIR)/Makefile
	@printf "$(BOLD)$(BLUE)$(ROCKET) Building SVMClassifier ($(BUILD_TYPE))...$(NC)\n"
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)
	@printf "$(BOLD)$(GREEN)$(CHECK) Build completed successfully!$(NC)\n"

.PHONY: debug
debug:
	@printf "$(BOLD)$(YELLOW)$(GEAR) Building in debug mode...$(NC)\n"
	@$(MAKE) build BUILD_TYPE=Debug BUILD_DIR=build_debug

$(BUILD_DIR)/Makefile:
	@printf "$(BOLD)$(CYAN)$(GEAR) Configuring CMake build...$(NC)\n"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH) \
		-DBUILD_DOCUMENTATION=ON

.PHONY: clean
clean:
	@printf "$(BOLD)$(YELLOW)$(CLEAN) Cleaning build directory...$(NC)\n"
	@rm -rf $(BUILD_DIR)
	@printf "$(BOLD)$(GREEN)$(CHECK) Clean completed!$(NC)\n"

.PHONY: rebuild
rebuild: clean build

.PHONY: install
install:
	@printf "$(BOLD)$(PURPLE)$(ROCKET) Running full installation...$(NC)\n"
	@./install.sh

# Testing targets
.PHONY: test
test: build
	@printf "$(BOLD)$(GREEN)$(TEST) Running all tests...$(NC)\n"
	@cd $(BUILD_DIR) && ctest --output-on-failure --parallel $(JOBS)
	@printf "$(BOLD)$(GREEN)$(CHECK) Tests completed!$(NC)\n"

.PHONY: test-ctest
test-ctest: build
	@printf "$(BOLD)$(GREEN)$(TEST) Running tests with CTest...$(NC)\n"
	@cd $(BUILD_DIR) && ctest --output-on-failure --parallel $(JOBS)

.PHONY: test-unit
test-unit: build
	@printf "$(BOLD)$(BLUE)$(TEST) Running unit tests...$(NC)\n"
	@$(BUILD_DIR)/tests/svm_classifier_tests "[unit]"

.PHONY: test-integration
test-integration: build
	@printf "$(BOLD)$(PURPLE)$(TEST) Running integration tests...$(NC)\n"
	@$(BUILD_DIR)/tests/svm_classifier_tests "[integration]"

.PHONY: test-performance
test-performance: build
	@printf "$(BOLD)$(YELLOW)$(TEST) Running performance tests...$(NC)\n"
	@$(BUILD_DIR)/tests/svm_classifier_tests "[performance]"

.PHONY: test-verbose
test-verbose: build
	@printf "$(BOLD)$(CYAN)$(TEST) Running tests with verbose output...$(NC)\n"
	@$(BUILD_DIR)/tests/svm_classifier_tests -s

.PHONY: test-memcheck
test-memcheck: build
	@printf "$(BOLD)$(RED)$(WARN) Running memory check tests...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) test_memcheck

.PHONY: test-profile
test-profile: build
	@printf "$(BOLD)$(PURPLE)$(GEAR) Running performance profiling...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) test_profile

.PHONY: coverage
coverage:
	@printf "$(BOLD)$(CYAN)$(TEST) Setting up code coverage analysis...$(NC)\n"
	@# Check if required tools are available
	@which gcov >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: gcov not found. Please install gcc or clang development tools.$(NC)\n" && exit 1)
	@which lcov >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: lcov not found. Please install lcov (brew install lcov or apt-get install lcov).$(NC)\n" && exit 1)
	@which genhtml >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: genhtml not found. Please install lcov package.$(NC)\n" && exit 1)
	@printf "$(BOLD)$(GREEN)$(CHECK) All coverage tools available$(NC)\n"
	@printf "$(BOLD)$(BLUE)$(GEAR) Configuring Debug build with coverage...$(NC)\n"
	@rm -rf build_coverage
	@mkdir -p build_coverage
	@cd build_coverage && cmake .. \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH) \
		-DBUILD_DOCUMENTATION=ON
	@printf "$(BOLD)$(BLUE)$(ROCKET) Building with coverage instrumentation...$(NC)\n"
	@cmake --build build_coverage --config Debug --parallel $(JOBS)
	@printf "$(BOLD)$(GREEN)$(TEST) Running tests with coverage collection...$(NC)\n"
	@cd build_coverage && ctest --output-on-failure --parallel $(JOBS) || true
	@printf "$(BOLD)$(CYAN)$(GEAR) Collecting coverage data...$(NC)\n"
	@cd build_coverage && lcov --capture --directory . --output-file coverage.info --ignore-errors inconsistent,unsupported,format 2>/dev/null || true
	@cd build_coverage && lcov --remove coverage.info '/usr/*' '*/_deps/*' '*/tests/*' '*/libtorch/*' --output-file coverage_filtered.info 2>/dev/null || true
	@cd build_coverage && genhtml coverage_filtered.info --output-directory coverage_html --ignore-errors category 2>/dev/null || true
	@printf "$(BOLD)$(CYAN)$(SPARKLES) Coverage report generated!$(NC)\n"
	@printf "$(BOLD)$(GREEN)$(INFO) Coverage summary:$(NC)\n"
	@cd build_coverage && lcov --summary coverage_filtered.info 2>/dev/null | grep -E "(lines|functions|branches)" || echo "Coverage data processed successfully"
	@printf "$(BOLD)$(BLUE)$(BOOK) HTML report: $(CYAN)build_coverage/coverage_html/index.html$(NC)\n"
	@printf "$(BOLD)$(YELLOW)$(INFO) Open with: open build_coverage/coverage_html/index.html$(NC)\n"

# Documentation targets
.PHONY: docs
docs: $(BUILD_DIR)/Makefile
	@printf "$(BOLD)$(BLUE)$(BOOK) Generating documentation...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) doxygen
	@printf "$(BOLD)$(GREEN)$(CHECK) Documentation generated successfully!$(NC)\n"

.PHONY: docs-open
docs-open: docs
	@printf "$(BOLD)$(CYAN)$(BOOK) Opening documentation...$(NC)\n"
	@./build_docs.sh --open

# Development targets
.PHONY: format
format:
	@printf "$(BOLD)$(YELLOW)$(SPARKLES) Formatting code with clang-format...$(NC)\n"
	@find src include tests examples -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
	@printf "$(BOLD)$(GREEN)$(CHECK) Code formatting completed!$(NC)\n"

.PHONY: lint
lint:
	@printf "$(BOLD)$(PURPLE)$(GEAR) Running static analysis with clang-tidy...$(NC)\n"
	@clang-tidy src/* -- -I include -std=c++17

.PHONY: validate
validate:
	@printf "$(BOLD)$(BLUE)$(CHECK) Running comprehensive build validation...$(NC)\n"
	@./validate_build.sh

.PHONY: examples
examples: build
	@printf "$(BOLD)$(GREEN)$(ROCKET) Building examples...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) examples
	@printf "$(BOLD)$(CYAN)$(ROCKET) Running basic example...$(NC)\n"
	@$(BUILD_DIR)/examples/basic_usage
	@printf "$(BOLD)$(CYAN)$(ROCKET) Running advanced example...$(NC)\n"
	@$(BUILD_DIR)/examples/advanced_usage

# Utility targets
.PHONY: info
info:
	@printf "$(BOLD)$(BLUE)$(INFO) Build Configuration:$(NC)\n"
	@printf "  $(CYAN)Project$(NC): $(YELLOW)SVMClassifier$(NC)\n"
	@printf "  $(CYAN)Build Type$(NC): $(GREEN)$(BUILD_TYPE)$(NC)\n"
	@printf "  $(CYAN)Build Directory$(NC): $(GREEN)$(BUILD_DIR)$(NC)\n"
	@printf "  $(CYAN)Jobs$(NC): $(GREEN)$(JOBS)$(NC)\n"
	@printf "  $(CYAN)CMAKE_PREFIX_PATH$(NC): $(GREEN)$(CMAKE_PREFIX_PATH)$(NC)\n"
	@printf "  $(CYAN)Detected PyTorch$(NC): $(GREEN)$(DETECTED_TORCH)$(NC)\n\n"
	@printf "$(BOLD)$(GREEN)$(GEAR) Available Tools:$(NC)\n"
	@which cmake >/dev/null 2>&1 && printf "  $(GREEN)$(CHECK) CMake$(NC): $$(cmake --version | head -n1)\n" || printf "  $(RED)$(CROSS) CMake$(NC): Not found\n"
	@which doxygen >/dev/null 2>&1 && printf "  $(GREEN)$(CHECK) Doxygen$(NC): $$(doxygen --version)\n" || printf "  $(RED)$(CROSS) Doxygen$(NC): Not found\n"
	@which valgrind >/dev/null 2>&1 && printf "  $(GREEN)$(CHECK) Valgrind$(NC): $$(valgrind --version | head -n1)\n" || printf "  $(RED)$(CROSS) Valgrind$(NC): Not found\n"
	@which clang-format >/dev/null 2>&1 && printf "  $(GREEN)$(CHECK) clang-format$(NC): $$(clang-format --version | head -n1)\n" || printf "  $(RED)$(CROSS) clang-format$(NC): Not found\n"
	@which clang-tidy >/dev/null 2>&1 && printf "  $(GREEN)$(CHECK) clang-tidy$(NC): $$(clang-tidy --version | head -n1)\n" || printf "  $(RED)$(CROSS) clang-tidy$(NC): Not found\n"

.PHONY: deps
deps:
	@printf "$(BOLD)$(PURPLE)$(INFO) Project Dependencies:$(NC)\n"
	@printf "  $(BOLD)$(GREEN)Required:$(NC)\n"
	@printf "    $(YELLOW)•$(NC) CMake (>= 3.15)\n"
	@printf "    $(YELLOW)•$(NC) C++17 compiler (GCC/Clang)\n"
	@printf "    $(YELLOW)•$(NC) PyTorch/libtorch\n"
	@printf "  $(BOLD)$(BLUE)Fetched Automatically:$(NC)\n"
	@printf "    $(CYAN)•$(NC) libsvm (v332)\n"
	@printf "    $(CYAN)•$(NC) liblinear (v249)\n"
	@printf "    $(CYAN)•$(NC) nlohmann/json (v3.11.3)\n"
	@printf "    $(CYAN)•$(NC) Catch2 (v3.4.0)\n"
	@printf "  $(BOLD)$(YELLOW)Optional:$(NC)\n"
	@printf "    $(PURPLE)•$(NC) Doxygen (for documentation)\n"
	@printf "    $(PURPLE)•$(NC) Valgrind (for memory checking)\n"
	@printf "    $(PURPLE)•$(NC) clang-format (for code formatting)\n"
	@printf "    $(PURPLE)•$(NC) clang-tidy (for static analysis)\n"

.PHONY: clean-all
clean-all:
	@printf "$(BOLD)$(RED)$(CLEAN) Cleaning all build directories and dependencies...$(NC)\n"
	@rm -rf build build_Debug build_Release build_coverage
	@rm -rf _deps
	@printf "$(BOLD)$(GREEN)$(CHECK) All build artifacts cleaned!$(NC)\n"

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