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

BUILD_DEBUG := build_debug
BUILD_RELEASE := build_release

# Configuration
BUILD_DIR := ${BUILD_RELEASE}
BUILD_TYPE := Release
COVERAGE := OFF
# Set the number of parallel jobs to the number of available processors minus 7
CPUS := $(shell getconf _NPROCESSORS_ONLN 2>/dev/null \
                 || nproc --all 2>/dev/null \
                 || sysctl -n hw.ncpu)
JOBS := $(shell n=$(CPUS); [ $${n} -gt 7 ] && echo $$((n-7)) || echo 1)


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
	@printf "$(BOLD)$(GREEN)$(ROCKET) Conan Packaging:$(NC)\n"
	@printf "  $(YELLOW)conan-create$(NC)  - $(ROCKET) Create Conan package (auto-extracts version)\n"
	@printf "  $(YELLOW)conan-export$(NC)  - $(GEAR) Export Conan recipe to local cache\n"
	@printf "  $(YELLOW)conan-install$(NC) - $(GEAR) Install dependencies with Conan\n"
	@printf "  $(YELLOW)conan-build$(NC)   - $(ROCKET) Build with Conan\n"
	@printf "  $(YELLOW)conan-test$(NC)    - $(TEST) Test Conan package\n\n"
	@printf "$(BOLD)$(PURPLE)$(GEAR) Configuration:$(NC)\n"
	@printf "  $(CYAN)BUILD_TYPE$(NC)=$(GREEN)$(BUILD_TYPE)$(NC)\n"
	@printf "  $(CYAN)JOBS$(NC)=$(GREEN)$(JOBS)$(NC)\n"

# Build targets
.PHONY: build
build: $(BUILD_DIR)/Makefile
	@printf "$(BOLD)$(BLUE)$(ROCKET) Building SVMClassifier ($(BUILD_TYPE))...$(NC)\n"
	@cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel $(JOBS)
	@printf "$(BOLD)$(GREEN)$(CHECK) Build completed successfully!$(NC)\n"

$(BUILD_DEBUG)/Makefile:
	@$(MAKE) debug

.PHONY: debug
debug:
	@printf "$(BOLD)$(YELLOW)$(GEAR) Building in debug mode...$(NC)\n"
	@$(MAKE) build BUILD_TYPE=Debug BUILD_DIR=$(BUILD_DEBUG) COVERAGE=ON

$(BUILD_DIR)/Makefile:
	@printf "$(BOLD)$(CYAN)$(GEAR) Configuring CMake build...$(NC)\n"
	@rm -fr $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	@conan install . --build=missing -of $(BUILD_DIR) -s build_type=$(BUILD_TYPE)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_TOOLCHAIN_FILE=$(BUILD_DIR)/build/$(BUILD_TYPE)/generators/conan_toolchain.cmake -DBUILD_DOCUMENTATION=ON -DENABLE_COVERAGE=$(COVERAGE)

.PHONY: clean
clean:
	@printf "$(BOLD)$(YELLOW)$(CLEAN) Cleaning build directories...$(NC)\n"
	@rm -rf $(BUILD_RELEASE) $(BUILD_DEBUG)
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
coverage: $(BUILD_DEBUG)/Makefile
	@printf "$(BOLD)$(CYAN)$(TEST) Setting up code coverage analysis...$(NC)\n"
	@# Check if required tools are available
	@which gcov >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: gcov not found. Please install gcc or clang development tools.$(NC)\n" && exit 1)
	@which lcov >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: lcov not found. Please install lcov (brew install lcov or apt-get install lcov).$(NC)\n" && exit 1)
	@which genhtml >/dev/null 2>&1 || (printf "$(BOLD)$(RED)$(CROSS) Error: genhtml not found. Please install lcov package.$(NC)\n" && exit 1)
	@printf "$(BOLD)$(GREEN)$(CHECK) All coverage tools available$(NC)\n"
	@printf "$(BOLD)$(BLUE)$(GEAR) Configuring Debug build with coverage...$(NC)\n"
	@printf "$(BOLD)$(BLUE)$(ROCKET) Building with coverage instrumentation...$(NC)\n"
	@printf "$(BOLD)$(GREEN)$(TEST) Running tests with coverage collection...$(NC)\n"
	@cd $(BUILD_DEBUG) && ulimit -s 8192 && ctest --output-on-failure --parallel $(JOBS)
	@printf "$(BOLD)$(CYAN)$(GEAR) Collecting coverage data...$(NC)\n"
	@cd $(BUILD_DEBUG) && lcov --capture --directory . --base-directory .. --output-file coverage.info --ignore-errors inconsistent,unsupported,format,source
	@cd $(BUILD_DEBUG) && lcov --remove coverage.info '/usr/*' '*/tests/*' '*/_deps/*' '*/.conan2/*' --output-file coverage_filtered.info --ignore-errors inconsistent,source
	@cd $(BUILD_DEBUG) && genhtml coverage_filtered.info --output-directory coverage_html --ignore-errors category,inconsistent,source
	@printf "$(BOLD)$(CYAN)$(SPARKLES) Coverage report generated!$(NC)\n"
	@lcov -l $(BUILD_DEBUG)/coverage_filtered.info --ignore-errors inconsistent,source
	@printf "$(BOLD)$(GREEN)$(INFO) Coverage summary:$(NC)\n"
	@cd $(BUILD_DEBUG) && lcov --summary coverage_filtered.info --ignore-errors inconsistent,source | grep -E "(lines|functions|branches)"
	@printf "$(BOLD)$(BLUE)$(BOOK) HTML report: $(CYAN)$(BUILD_DEBUG)/coverage_html/index.html$(NC)\n"
	@printf "$(BOLD)$(YELLOW)$(INFO) Open with: open/xdg-open $(BUILD_DEBUG)/coverage_html/index.html$(NC)\n"

# Documentation targets
.PHONY: docs
docs: $(BUILD_DIR)/Makefile
	@printf "$(BOLD)$(BLUE)$(BOOK) Generating documentation...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) doxygen
	@printf "$(BOLD)$(GREEN)$(CHECK) Documentation generated successfully!$(NC)\n"

.PHONY: docs-open
docs-open: $(BUILD_DIR)/Makefile
	@printf "$(BOLD)$(CYAN)$(BOOK) Opening documentation...$(NC)\n"
	@cd $(BUILD_DIR) && $(MAKE) open_docs

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
	@rm -rf build $(BUILD_DEBUG)
	@rm -rf _deps
	@printf "$(BOLD)$(GREEN)$(CHECK) All build artifacts cleaned!$(NC)\n"

# Conan packaging targets
.PHONY: conan-create
conan-create:
	@printf "$(BOLD)$(PURPLE)$(ROCKET) Creating Conan package...$(NC)\n"
	@printf "$(BOLD)$(CYAN)$(INFO) Version will be extracted from CMakeLists.txt$(NC)\n"
	@conan create . --build=missing
	@printf "$(BOLD)$(GREEN)$(CHECK) Conan package created successfully!$(NC)\n"

.PHONY: conan-export
conan-export:
	@printf "$(BOLD)$(BLUE)$(GEAR) Exporting Conan recipe...$(NC)\n"
	@conan export .
	@printf "$(BOLD)$(GREEN)$(CHECK) Conan recipe exported successfully!$(NC)\n"

.PHONY: conan-install
conan-install:
	@printf "$(BOLD)$(CYAN)$(GEAR) Installing dependencies with Conan...$(NC)\n"
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && conan install .. --build=missing
	@printf "$(BOLD)$(GREEN)$(CHECK) Conan dependencies installed!$(NC)\n"

.PHONY: conan-build
conan-build:
	@printf "$(BOLD)$(YELLOW)$(ROCKET) Building with Conan...$(NC)\n"
	@conan build .
	@printf "$(BOLD)$(GREEN)$(CHECK) Conan build completed!$(NC)\n"

.PHONY: conan-test
conan-test:
	@printf "$(BOLD)$(GREEN)$(TEST) Testing Conan package...$(NC)\n"
	@conan test test_package/conanfile.py svmclassifier
	@printf "$(BOLD)$(GREEN)$(CHECK) Conan package test passed!$(NC)\n"

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

.PHONY: package
package: conan-create

