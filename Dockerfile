# Multi-stage Dockerfile for SVMClassifier
FROM ubuntu:22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_BUILD_TYPE=Release

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    python3 \
    python3-pip \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch C++ (libtorch)
WORKDIR /opt
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip \
    && rm libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# Set PyTorch environment
ENV Torch_DIR=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Create build directory
WORKDIR /workspace
COPY . .

# Build the project
RUN mkdir build && cd build \
    && cmake .. \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_PREFIX_PATH=/opt/libtorch \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j$(nproc) \
    && make test \
    && make install

# Runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy libtorch libraries
COPY --from=builder /opt/libtorch/lib /usr/local/lib/
COPY --from=builder /usr/local /usr/local/

# Update library cache
RUN ldconfig

# Create non-root user
RUN useradd -m -s /bin/bash svmuser
USER svmuser
WORKDIR /home/svmuser

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Default command
CMD ["bash"]

# Development stage (includes build tools and source)
FROM builder AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    clang-format \
    clang-tidy \
    doxygen \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install code coverage tools
RUN apt-get update && apt-get install -y \
    gcov \
    lcov \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Default command for development
CMD ["bash"]