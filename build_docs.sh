#!/bin/bash

# Documentation Build Script for SVM Classifier C++
# This script builds the API documentation using Doxygen

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
OPEN_DOCS=false
CLEAN_DOCS=false
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Documentation Build Script for SVM Classifier C++

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -b, --build-dir     Build directory (default: build)
    -o, --open          Open documentation in browser after build
    -c, --clean         Clean documentation before building
    -v, --verbose       Enable verbose output

EXAMPLES:
    $0                  # Build documentation
    $0 --open           # Build and open in browser
    $0 --clean --open   # Clean, build, and open

REQUIREMENTS:
    - Doxygen must be installed
    - Project must be configured with CMake
    - Graphviz (optional, for enhanced diagrams)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -o|--open)
            OPEN_DOCS=true
            shift
            ;;
        -c|--clean)
            CLEAN_DOCS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
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

print_info "Building SVM Classifier C++ Documentation"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -f "docs/Doxyfile.in" ]; then
    print_error "Please run this script from the SVMClassifier root directory"
    print_error "Missing: CMakeLists.txt or docs/Doxyfile.in"
    exit 1
fi

# Check if Doxygen is available
if ! command -v doxygen >/dev/null 2>&1; then
    print_error "Doxygen not found. Please install Doxygen to build documentation."
    print_info "On Ubuntu/Debian: sudo apt-get install doxygen"
    print_info "On macOS: brew install doxygen"
    print_info "On Windows: choco install doxygen.install"
    exit 1
fi

DOXYGEN_VERSION=$(doxygen --version)
print_info "Using Doxygen version: $DOXYGEN_VERSION"

# Check for Graphviz (optional)
if command -v dot >/dev/null 2>&1; then
    DOT_VERSION=$(dot -V 2>&1 | head -1)
    print_info "Graphviz found: $DOT_VERSION"
    print_info "Enhanced diagrams will be generated"
else
    print_warning "Graphviz not found. Basic diagrams only."
    print_info "Install Graphviz for enhanced class diagrams and graphs"
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    print_error "Build directory '$BUILD_DIR' not found"
    print_info "Please run CMake configuration first:"
    print_info "  mkdir $BUILD_DIR && cd $BUILD_DIR"
    print_info "  cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch"
    exit 1
fi

# Check if CMake has been configured
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    print_error "CMake not configured in '$BUILD_DIR'"
    print_info "Please run CMake configuration first:"
    print_info "  cd $BUILD_DIR && cmake .."
    exit 1
fi

# Clean documentation if requested
if [ "$CLEAN_DOCS" = true ]; then
    print_info "Cleaning existing documentation..."
    rm -rf "$BUILD_DIR/docs"
    print_success "Documentation cleaned"
fi

# Build documentation
print_info "Building documentation..."

cd "$BUILD_DIR"

# Check if doxygen target is available
if cmake --build . --target help 2>/dev/null | grep -q "doxygen"; then
    print_info "Using CMake doxygen target"
    
    if cmake --build . --target doxygen; then
        print_success "Documentation built successfully!"
        
        # Check if documentation was actually generated
        if [ -f "docs/html/index.html" ]; then
            DOC_SIZE=$(du -sh docs/html 2>/dev/null | cut -f1)
            print_success "Documentation size: $DOC_SIZE"
            
            # Count number of HTML files generated
            HTML_COUNT=$(find docs/html -name "*.html" | wc -l)
            print_info "Generated $HTML_COUNT HTML pages"
            
            # Check for warnings
            if [ -f "docs/doxygen_warnings.log" ] && [ -s "docs/doxygen_warnings.log" ]; then
                WARNING_COUNT=$(wc -l < docs/doxygen_warnings.log)
                print_warning "Documentation has $WARNING_COUNT warnings"
                
                if [ "$VERBOSE" = true ]; then
                    print_info "Recent warnings:"
                    tail -5 docs/doxygen_warnings.log | while read -r line; do
                        print_warning "  $line"
                    done
                fi
            else
                print_success "No warnings generated"
            fi
            
        else
            print_error "Documentation index file not found"
            exit 1
        fi
    else
        print_error "Documentation build failed"
        exit 1
    fi
else
    print_error "Doxygen target not available"
    print_info "Make sure Doxygen was found during CMake configuration"
    print_info "Reconfigure with: cmake .. -DBUILD_DOCUMENTATION=ON"
    exit 1
fi

cd ..

# Open documentation if requested
if [ "$OPEN_DOCS" = true ]; then
    DOC_INDEX="$BUILD_DIR/docs/html/index.html"
    
    if [ -f "$DOC_INDEX" ]; then
        print_info "Opening documentation in browser..."
        
        # Detect platform and open browser
        if command -v xdg-open >/dev/null 2>&1; then
            # Linux
            xdg-open "$DOC_INDEX"
        elif command -v open >/dev/null 2>&1; then
            # macOS
            open "$DOC_INDEX"
        elif command -v start >/dev/null 2>&1; then
            # Windows
            start "$DOC_INDEX"
        else
            print_warning "Could not detect browser. Please open manually:"
            print_info "file://$(realpath "$DOC_INDEX")"
        fi
    else
        print_error "Documentation index not found: $DOC_INDEX"
    fi
fi

print_success "Documentation build completed!"
print_info "Documentation location: $BUILD_DIR/docs/html/"
print_info "Main page: $BUILD_DIR/docs/html/index.html"

# Provide helpful next steps
echo
print_info "Next steps:"
print_info "  - Open docs/html/index.html in a web browser"
print_info "  - Share the docs/ directory for deployment"
print_info "  - Use 'cmake --build $BUILD_DIR --target doxygen' to rebuild"

exit 0
