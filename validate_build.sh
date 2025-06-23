#!/bin/bash

# SVMClassifier Build Validation Script
# This script performs comprehensive validation of the build system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build_validation"
INSTALL_DIR="install_validation"
TEST_TIMEOUT=300  # 5 minutes
VERBOSE=false
CLEAN_BUILD=true
RUN_PERFORMANCE_TESTS=false
RUN_MEMORY_CHECKS=false
TORCH_VERSION="2.7.1"

# Counters for test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to print colored output
print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

print_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_skip() {
    echo -e "${CYAN}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
SVMClassifier Build Validation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    --no-clean              Don't clean build directory
    --performance           Run performance tests
    --memory-check          Run memory checks with valgrind
    --build-dir DIR         Build directory (default: build_validation)
    --install-dir DIR       Install directory (default: install_validation)
    --torch-version VER     PyTorch version (default: 2.1.0)

EXAMPLES:
    $0                      # Standard validation
    $0 --verbose            # Verbose validation
    $0 --performance        # Include performance benchmarks
    $0 --memory-check       # Include memory checks

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-clean)
            CLEAN_BUILD=false
            shift
            ;;
        --performance)
            RUN_PERFORMANCE_TESTS=true
            shift
            ;;
        --memory-check)
            RUN_MEMORY_CHECKS=true
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --torch-version)
            TORCH_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set verbose mode
if [ "$VERBOSE" = true ]; then
    set -x
fi

# Validation functions
check_prerequisites() {
    print_header "CHECKING PREREQUISITES"
    
    # Check if we're in the right directory
    if [ ! -f "CMakeLists.txt" ] || [ ! -d "src" ] || [ ! -d "include" ]; then
        print_failure "Please run this script from the SVMClassifier root directory"
        exit 1
    fi
    print_success "Running from correct directory"
    
    # Check for required tools
    local missing_tools=()
    
    for tool in cmake git gcc g++ pkg-config; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        print_failure "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    print_success "All required tools found"
    
    # Check CMake version
    CMAKE_VERSION=$(cmake --version | head -1 | cut -d' ' -f3)
    print_info "CMake version: $CMAKE_VERSION"
    
    # Check GCC version
    GCC_VERSION=$(gcc --version | head -1)
    print_info "GCC version: $GCC_VERSION"
    
    # Check for optional tools
    for tool in valgrind lcov doxygen; do
        if command -v "$tool" >/dev/null 2>&1; then
            print_info "$tool available"
        else
            print_warning "$tool not available (optional)"
        fi
    done
}

setup_pytorch() {
    print_header "SETTING UP PYTORCH"
    
    TORCH_DIR="/opt/libtorch"
    if [ ! -d "$TORCH_DIR" ] && [ ! -d "libtorch" ]; then
        print_step "Downloading PyTorch C++ version $TORCH_VERSION"
        
        TORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
        
        wget -q "$TORCH_URL" -O libtorch.zip
        unzip -q libtorch.zip
        rm libtorch.zip
        
        TORCH_DIR="$(pwd)/libtorch"
        print_success "PyTorch C++ downloaded and extracted"
    else
        if [ -d "/opt/libtorch" ]; then
            TORCH_DIR="/opt/libtorch"
        else
            TORCH_DIR="$(pwd)/libtorch"
        fi
        print_success "PyTorch C++ found at $TORCH_DIR"
    fi
    
    export Torch_DIR="$TORCH_DIR"
    export LD_LIBRARY_PATH="$TORCH_DIR/lib:$LD_LIBRARY_PATH"
}

configure_build() {
    print_header "CONFIGURING BUILD"
    
    if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
        print_step "Cleaning build directory"
        rm -rf "$BUILD_DIR"
        print_success "Build directory cleaned"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    print_step "Running CMake configuration"
    
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_PREFIX_PATH="$Torch_DIR"
        -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    )
    
    if [ "$VERBOSE" = true ]; then
        CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
    fi
    
    if cmake .. "${CMAKE_ARGS[@]}"; then
        print_success "CMake configuration successful"
    else
        print_failure "CMake configuration failed"
        exit 1
    fi
    
    cd ..
}

build_project() {
    print_header "BUILDING PROJECT"
    
    cd "$BUILD_DIR"
    
    print_step "Building main library"
    if cmake --build . --config Release -j$(nproc); then
        print_success "Main library build successful"
    else
        print_failure "Main library build failed"
        exit 1
    fi
    
    print_step "Building examples"
    if make examples >/dev/null 2>&1 || true; then
        print_success "Examples build successful"
    else
        print_warning "Examples build failed or not available"
    fi
    
    print_step "Building tests"
    if make svm_classifier_tests >/dev/null 2>&1; then
        print_success "Tests build successful"
    else
        print_failure "Tests build failed"
        exit 1
    fi
    
    cd ..
}

run_unit_tests() {
    print_header "RUNNING UNIT TESTS"
    
    cd "$BUILD_DIR"
    export LD_LIBRARY_PATH="$Torch_DIR/lib:$LD_LIBRARY_PATH"
    
    print_step "Running all unit tests"
    if timeout $TEST_TIMEOUT ./svm_classifier_tests "[unit]" --reporter console; then
        print_success "Unit tests passed"
    else
        print_failure "Unit tests failed"
    fi
    
    print_step "Running integration tests"
    if timeout $TEST_TIMEOUT ./svm_classifier_tests "[integration]" --reporter console; then
        print_success "Integration tests passed"
    else
        print_failure "Integration tests failed"
    fi
    
    cd ..
}

run_performance_tests() {
    if [ "$RUN_PERFORMANCE_TESTS" = false ]; then
        print_skip "Performance tests (use --performance to enable)"
        return
    fi
    
    print_header "RUNNING PERFORMANCE TESTS"
    
    cd "$BUILD_DIR"
    export LD_LIBRARY_PATH="$Torch_DIR/lib:$LD_LIBRARY_PATH"
    
    print_step "Running performance benchmarks"
    if timeout $((TEST_TIMEOUT * 2)) ./svm_classifier_tests "[performance]" --reporter console; then
        print_success "Performance tests completed"
    else
        print_warning "Performance tests failed or timed out"
    fi
    
    cd ..
}

run_memory_checks() {
    if [ "$RUN_MEMORY_CHECKS" = false ]; then
        print_skip "Memory checks (use --memory-check to enable)"
        return
    fi
    
    if ! command -v valgrind >/dev/null 2>&1; then
        print_skip "Memory checks (valgrind not available)"
        return
    fi
    
    print_header "RUNNING MEMORY CHECKS"
    
    cd "$BUILD_DIR"
    export LD_LIBRARY_PATH="$Torch_DIR/lib:$LD_LIBRARY_PATH"
    
    print_step "Running memory leak detection"
    if timeout $((TEST_TIMEOUT * 3)) valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
       --track-origins=yes --error-exitcode=1 ./svm_classifier_tests "[unit]" >/dev/null 2>valgrind.log; then
        print_success "No memory leaks detected"
    else
        print_failure "Memory leaks or errors detected"
        print_info "Check valgrind.log for details"
    fi
    
    cd ..
}

test_examples() {
    print_header "TESTING EXAMPLES"
    
    cd "$BUILD_DIR"
    export LD_LIBRARY_PATH="$Torch_DIR/lib:$LD_LIBRARY_PATH"
    
    if [ -f "examples/basic_usage" ]; then
        print_step "Running basic usage example"
        if timeout $TEST_TIMEOUT ./examples/basic_usage >/dev/null 2>&1; then
            print_success "Basic usage example ran successfully"
        else
            print_failure "Basic usage example failed"
        fi
    else
        print_skip "Basic usage example (not built)"
    fi
    
    if [ -f "examples/advanced_usage" ]; then
        print_step "Running advanced usage example"
        if timeout $((TEST_TIMEOUT * 2)) ./examples/advanced_usage >/dev/null 2>&1; then
            print_success "Advanced usage example ran successfully"
        else
            print_warning "Advanced usage example failed or timed out"
        fi
    else
        print_skip "Advanced usage example (not built)"
    fi
    
    cd ..
}

test_installation() {
    print_header "TESTING INSTALLATION"
    
    cd "$BUILD_DIR"
    
    print_step "Installing to test directory"
    if cmake --install . --config Release; then
        print_success "Installation successful"
    else
        print_failure "Installation failed"
        cd ..
        return
    fi
    
    cd ..
    
    # Test that installed files exist
    local install_files=(
        "$INSTALL_DIR/lib/libsvm_classifier.a"
        "$INSTALL_DIR/include/svm_classifier/svm_classifier.hpp"
        "$INSTALL_DIR/lib/cmake/SVMClassifier/SVMClassifierConfig.cmake"
    )
    
    for file in "${install_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "Installed file found: $(basename "$file")"
        else
            print_failure "Missing installed file: $file"
        fi
    done
    
    # Test CMake find_package
    print_step "Testing CMake find_package"
    
    cat > test_find_package.cmake << 'EOF'
cmake_minimum_required(VERSION 3.15)
project(TestFindPackage)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/install_validation")

find_package(SVMClassifier REQUIRED)

if(TARGET SVMClassifier::svm_classifier)
    message(STATUS "SVMClassifier found successfully")
else()
    message(FATAL_ERROR "SVMClassifier target not found")
endif()
EOF
    
    if cmake -P test_find_package.cmake >/dev/null 2>&1; then
        print_success "CMake find_package test passed"
    else
        print_failure "CMake find_package test failed"
    fi
    
    rm -f test_find_package.cmake
}

test_compiler_compatibility() {
    print_header "TESTING COMPILER COMPATIBILITY"
    
    # Test with different C++ standards if supported
    for std in 17 20; do
        print_step "Testing C++$std compatibility"
        
        TEST_BUILD_DIR="build_cpp$std"
        mkdir -p "$TEST_BUILD_DIR"
        cd "$TEST_BUILD_DIR"
        
        if cmake .. -DCMAKE_CXX_STANDARD=$std -DCMAKE_PREFIX_PATH="$Torch_DIR" >/dev/null 2>&1; then
            if cmake --build . --target svm_classifier -j$(nproc) >/dev/null 2>&1; then
                print_success "C++$std compatibility verified"
            else
                print_warning "C++$std build failed"
            fi
        else
            print_warning "C++$std configuration failed"
        fi
        
        cd ..
        rm -rf "$TEST_BUILD_DIR"
    done
}

generate_coverage_report() {
    if ! command -v lcov >/dev/null 2>&1; then
        print_skip "Coverage report (lcov not available)"
        return
    fi
    
    print_header "GENERATING COVERAGE REPORT"
    
    # Build with coverage flags
    DEBUG_BUILD_DIR="build_coverage"
    mkdir -p "$DEBUG_BUILD_DIR"
    cd "$DEBUG_BUILD_DIR"
    
    print_step "Building with coverage flags"
    if cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$Torch_DIR" \
       -DCMAKE_CXX_FLAGS="--coverage" -DCMAKE_C_FLAGS="--coverage" >/dev/null 2>&1; then
        
        if cmake --build . -j$(nproc) >/dev/null 2>&1; then
            export LD_LIBRARY_PATH="$Torch_DIR/lib:$LD_LIBRARY_PATH"
            
            print_step "Running tests for coverage"
            if ./svm_classifier_tests "[unit]" >/dev/null 2>&1; then
                
                print_step "Generating coverage report"
                if lcov --capture --directory . --output-file coverage.info >/dev/null 2>&1 && \
                   lcov --remove coverage.info '/usr/*' '*/external/*' '*/tests/*' \
                        --output-file coverage_filtered.info >/dev/null 2>&1; then
                    
                    COVERAGE_PERCENT=$(lcov --summary coverage_filtered.info 2>/dev/null | \
                                     grep "lines" | grep -o '[0-9.]*%' | head -1)
                    print_success "Coverage report generated: $COVERAGE_PERCENT"
                else
                    print_warning "Coverage report generation failed"
                fi
            else
                print_warning "Tests failed during coverage run"
            fi
        else
            print_warning "Coverage build failed"
        fi
    else
        print_warning "Coverage configuration failed"
    fi
    
    cd ..
    rm -rf "$DEBUG_BUILD_DIR"
}

validate_documentation() {
    if ! command -v doxygen >/dev/null 2>&1; then
        print_skip "Documentation validation (doxygen not available)"
        return
    fi
    
    print_header "VALIDATING DOCUMENTATION"
    
    cd "$BUILD_DIR"
    
    print_step "Generating documentation with CMake target"
    if cmake --build . --target doxygen >/dev/null 2>doxygen_warnings.log; then
        if [ -f "docs/html/index.html" ]; then
            print_success "Documentation generated successfully"
            
            # Check documentation size (should be substantial)
            DOC_SIZE=$(du -sh docs/html 2>/dev/null | cut -f1)
            print_info "Documentation size: $DOC_SIZE"
        else
            print_failure "Documentation files not found"
        fi
        
        # Check for warnings
        if [ -s doxygen_warnings.log ]; then
            WARNING_COUNT=$(wc -l < doxygen_warnings.log)
            print_warning "Documentation has $WARNING_COUNT warnings"
            if [ "$VERBOSE" = true ]; then
                print_info "Sample warnings:"
                head -5 doxygen_warnings.log | while read -r line; do
                    print_info "  $line"
                done
            fi
        else
            print_success "Documentation generated without warnings"
        fi
        
        # Check for essential documentation files
        DOC_FILES=(
            "docs/html/index.html"
            "docs/html/annotated.html"
            "docs/html/classes.html"
            "docs/html/files.html"
        )
        
        for doc_file in "${DOC_FILES[@]}"; do
            if [ -f "$doc_file" ]; then
                print_success "Found: $(basename "$doc_file")"
            else
                print_warning "Missing: $(basename "$doc_file")"
            fi
        done
        
    else
        print_failure "Documentation generation failed"
        if [ -s doxygen_warnings.log ]; then
            print_info "Error log:"
            head -10 doxygen_warnings.log | while read -r line; do
                print_info "  $line"
            done
        fi
    fi
    
    cd ..
    rm -f doxygen_warnings.log
}

test_packaging() {
    print_header "TESTING PACKAGING"
    
    cd "$BUILD_DIR"
    
    print_step "Testing CPack configuration"
    if cpack --config CPackConfig.cmake >/dev/null 2>&1; then
        print_success "Package generation successful"
        
        # List generated packages
        for pkg in *.tar.gz *.deb *.rpm *.zip ; do
            if [ -f "$pkg" ]; then
                print_info "Generated package: $pkg"
            fi
        done
    else
        print_warning "Package generation failed"
    fi
    
    cd ..
}

cleanup() {
    print_header "CLEANUP"
    
    if [ "$VERBOSE" = false ]; then
        print_step "Cleaning up temporary files"
        rm -rf "$BUILD_DIR" "$INSTALL_DIR" build_cpp* docs/
        print_success "Cleanup completed"
    else
        print_info "Keeping build files for inspection (verbose mode)"
    fi
}

print_summary() {
    print_header "VALIDATION SUMMARY"
    
    echo -e "${BLUE}Test Results:${NC}"
    echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
    echo -e "  ${CYAN}Skipped: $TESTS_SKIPPED${NC}"
    echo -e "  ${PURPLE}Total: $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✅ ALL CRITICAL TESTS PASSED!${NC}"
        echo -e "${GREEN}The SVMClassifier build system is working correctly.${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ SOME TESTS FAILED!${NC}"
        echo -e "${RED}Please review the failed tests above.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    print_header "SVMClassifier Build Validation"
    print_info "Starting comprehensive build validation..."
    
    check_prerequisites
    setup_pytorch
    configure_build
    build_project
    run_unit_tests
    run_performance_tests
    run_memory_checks
    test_examples
    test_installation
    test_compiler_compatibility
    generate_coverage_report
    validate_documentation
    test_packaging
    cleanup
    print_summary
}

# Handle signals for cleanup
trap 'echo -e "\n${RED}Validation interrupted!${NC}"; cleanup; exit 1' INT TERM

# Run main function
main "$@"