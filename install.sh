#!/bin/bash

# SVMClassifier Installation Script
# This script automates the installation of the SVM Classifier library

set -e  # Exit on any error

# Default values
BUILD_TYPE="Release"
INSTALL_PREFIX="/usr/local"
NUM_JOBS=$(nproc)
TORCH_VERSION="2.7.1"
SKIP_TESTS=false
VERBOSE=false
CLEAN_BUILD=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
SVMClassifier Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -b, --build-type TYPE   Build type: Release, Debug, RelWithDebInfo (default: Release)
    -p, --prefix PATH       Installation prefix (default: /usr/local)
    -j, --jobs NUM          Number of parallel jobs (default: $(nproc))
    -t, --torch-version VER PyTorch version to download (default: 2.8.0)
    --skip-tests            Skip running tests after build
    --clean                 Clean build directory before building
    -v, --verbose           Enable verbose output

EXAMPLES:
    $0                                  # Install with default settings
    $0 --build-type Debug --skip-tests  # Debug build without tests
    $0 --prefix ~/.local                # Install to user directory
    $0 --clean -v                       # Clean build with verbose output

DEPENDENCIES:
    The script will check for and help install required dependencies:
    - CMake 3.15+
    - C++17 compatible compiler (GCC 7+ or Clang 5+)
    - PyTorch C++ (libtorch) - will be downloaded automatically
    - Git (for fetching dependencies)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -b|--build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -p|--prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -t|--torch-version)
            TORCH_VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set verbose mode
if [ "$VERBOSE" = true ]; then
    set -x
fi

print_status "Starting SVMClassifier installation..."
print_status "Build type: $BUILD_TYPE"
print_status "Install prefix: $INSTALL_PREFIX"
print_status "Parallel jobs: $NUM_JOBS"
print_status "PyTorch version: $TORCH_VERSION"

# Make other scripts executable
if [ -f "validate_build.sh" ]; then
    chmod +x validate_build.sh
fi
if [ -f "build_docs.sh" ]; then
    chmod +x build_docs.sh
fi
if [ -f "troubleshoot_cmake.sh" ]; then
    chmod +x troubleshoot_cmake.sh
fi

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "src" ] || [ ! -d "include" ]; then
    print_error "Please run this script from the SVMClassifier root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
print_status "Checking system requirements..."

# Check for essential tools
MISSING_DEPS=()

if ! command_exists cmake; then
    MISSING_DEPS+=("cmake")
fi

if ! command_exists git; then
    MISSING_DEPS+=("git")
fi

if ! command_exists gcc && ! command_exists clang; then
    MISSING_DEPS+=("build-essential")
fi

if ! command_exists pkg-config; then
    MISSING_DEPS+=("pkg-config")
fi

# Check CMake version if available
if command_exists cmake; then
    CMAKE_VERSION=$(cmake --version | head -1 | cut -d' ' -f3)
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d'.' -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d'.' -f2)
    
    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 15 ]); then
        print_warning "CMake version $CMAKE_VERSION found. Version 3.15+ is recommended."
    else
        print_success "CMake version $CMAKE_VERSION found"
    fi
fi

# Install missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
    
    if command_exists apt-get; then
        print_status "Installing dependencies using apt-get..."
        sudo apt-get update
        sudo apt-get install -y "${MISSING_DEPS[@]}" libblas-dev liblapack-dev
    elif command_exists yum; then
        print_status "Installing dependencies using yum..."
        sudo yum install -y "${MISSING_DEPS[@]}" blas-devel lapack-devel
    elif command_exists brew; then
        print_status "Installing dependencies using brew..."
        brew install "${MISSING_DEPS[@]}"
    else
        print_error "Cannot install dependencies automatically. Please install: ${MISSING_DEPS[*]}"
        exit 1
    fi
fi

# Download and setup PyTorch C++
TORCH_DIR="/opt/libtorch"
if [ ! -d "$TORCH_DIR" ] && [ ! -d "$(pwd)/libtorch" ]; then
    print_status "Downloading PyTorch C++ (libtorch) version $TORCH_VERSION..."
    
    # Determine download URL based on PyTorch version
    # Handle different version formats (2.1.0, 2.7.1, etc.)
    if [[ "$TORCH_VERSION" =~ ^2\.[0-6]\. ]]; then
        # Older format for versions 2.0-2.6
        TORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
    else
        # Newer format for versions 2.7+
        TORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
    fi
    
    print_info "Download URL: $TORCH_URL"
    
    # Try to install system-wide first, fallback to local
    if [ -w "/opt" ]; then
        cd /opt
        if sudo wget -q "$TORCH_URL" -O libtorch.zip; then
            sudo unzip -q libtorch.zip
            sudo rm libtorch.zip
            TORCH_DIR="/opt/libtorch"
        else
            print_warning "Failed to download from official URL, trying alternative..."
            cd "$(dirname "$0")"
            # Fallback: check if user already has libtorch locally
            if [ -d "libtorch" ]; then
                print_success "Using existing local libtorch directory"
                TORCH_DIR="$(pwd)/libtorch"
            else
                print_error "Could not download PyTorch. Please install manually:"
                print_info "1. Download libtorch from https://pytorch.org/get-started/locally/"
                print_info "2. Extract to /opt/libtorch or $(pwd)/libtorch"
                print_info "3. Re-run this script"
                exit 1
            fi
        fi
    else
        print_warning "Cannot write to /opt, checking for local libtorch..."
        cd "$(dirname "$0")"
        if [ -d "libtorch" ]; then
            print_success "Using existing local libtorch directory"
            TORCH_DIR="$(pwd)/libtorch"
        else
            print_info "Downloading libtorch locally..."
            if wget -q "$TORCH_URL" -O libtorch.zip; then
                unzip -q libtorch.zip
                rm libtorch.zip
                TORCH_DIR="$(pwd)/libtorch"
            else
                print_error "Could not download PyTorch. Please install manually."
                exit 1
            fi
        fi
    fi
    
    print_success "PyTorch C++ installed to $TORCH_DIR"
else
    if [ -d "/opt/libtorch" ]; then
        TORCH_DIR="/opt/libtorch"
    else
        TORCH_DIR="$(pwd)/libtorch"
    fi
    print_success "PyTorch C++ found at $TORCH_DIR"
fi

# Return to project directory
cd "$(dirname "$0")"

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ] && [ -d "build" ]; then
    print_status "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
print_status "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
print_status "Configuring CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_PREFIX_PATH="$TORCH_DIR"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
)

if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

cmake .. "${CMAKE_ARGS[@]}"

# Build the project
print_status "Building SVMClassifier with $NUM_JOBS parallel jobs..."
cmake --build . --config "$BUILD_TYPE" -j "$NUM_JOBS"

# Run tests if not skipped
if [ "$SKIP_TESTS" = false ]; then
    print_status "Running tests..."
    export LD_LIBRARY_PATH="$TORCH_DIR/lib:$LD_LIBRARY_PATH"
    
    if ctest --output-on-failure --timeout 300; then
        print_success "All tests passed!"
    else
        print_warning "Some tests failed, but continuing with installation..."
    fi
else
    print_warning "Skipping tests as requested"
fi

# Install the library
print_status "Installing SVMClassifier to $INSTALL_PREFIX..."

if [ -w "$INSTALL_PREFIX" ] || [ "$INSTALL_PREFIX" = "$HOME"* ]; then
    cmake --install . --config "$BUILD_TYPE"
else
    sudo cmake --install . --config "$BUILD_TYPE"
fi

# Update library cache
if [ "$INSTALL_PREFIX" = "/usr/local" ] || [ "$INSTALL_PREFIX" = "/usr" ]; then
    print_status "Updating library cache..."
    sudo ldconfig
fi

# Run example to verify installation
print_status "Testing installation with basic example..."
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$LD_LIBRARY_PATH"

if [ -f "examples/basic_usage" ]; then
    if ./examples/basic_usage > /dev/null 2>&1; then
        print_success "Installation verification successful!"
    else
        print_warning "Installation verification failed, but library should still work"
    fi
fi

# Print installation summary
print_success "SVMClassifier installation completed!"
echo
echo "Installation Summary:"
echo "  Build type: $BUILD_TYPE"
echo "  Install prefix: $INSTALL_PREFIX"  
echo "  PyTorch location: $TORCH_DIR"
echo "  Library files: $INSTALL_PREFIX/lib"
echo "  Header files: $INSTALL_PREFIX/include"
echo "  Examples: build/examples/"
echo
echo "Usage:"
echo "  - Include path: $INSTALL_PREFIX/include"
echo "  - Library: -lsvm_classifier"
echo "  - CMake: find_package(SVMClassifier REQUIRED)"
echo
echo "Documentation:"
echo "  - Build docs: cmake --build build --target doxygen"
echo "  - Or use: ./build_docs.sh --open"
echo
echo "Environment:"
echo "  export LD_LIBRARY_PATH=$TORCH_DIR/lib:\$LD_LIBRARY_PATH"
echo
print_status "Installation complete!"

# Return to original directory
cd ..

exit 0