# External dependencies CMakeLists.txt

include(FetchContent)

# Fetch libsvm
FetchContent_Declare(
    libsvm
    GIT_REPOSITORY https://github.com/cjlin1/libsvm.git
    GIT_TAG v332
)
FetchContent_MakeAvailable(libsvm)

# Fetch liblinear
FetchContent_Declare(
    liblinear
    GIT_REPOSITORY https://github.com/cjlin1/liblinear.git
    GIT_TAG v249
)
FetchContent_MakeAvailable(liblinear)

# Build libsvm as static library
set(LIBSVM_SOURCES
    ${libsvm_SOURCE_DIR}/svm.cpp
)

add_library(libsvm_static STATIC ${LIBSVM_SOURCES})
target_include_directories(libsvm_static PUBLIC ${libsvm_SOURCE_DIR})
target_compile_definitions(libsvm_static PRIVATE -DLIBSVM_VERSION=332)

# Build liblinear as static library
set(LIBLINEAR_SOURCES
    ${liblinear_SOURCE_DIR}/linear.cpp
    ${liblinear_SOURCE_DIR}/tron.cpp
    ${liblinear_SOURCE_DIR}/blas/daxpy.c
    ${liblinear_SOURCE_DIR}/blas/ddot.c
    ${liblinear_SOURCE_DIR}/blas/dnrm2.c
    ${liblinear_SOURCE_DIR}/blas/dscal.c
)

add_library(liblinear_static STATIC ${LIBLINEAR_SOURCES})
target_include_directories(liblinear_static 
    PUBLIC 
        ${liblinear_SOURCE_DIR}
        ${liblinear_SOURCE_DIR}/blas
)
target_compile_definitions(liblinear_static PRIVATE -DLIBLINEAR_VERSION=249)

# Set C++ standard for the libraries
set_property(TARGET libsvm_static PROPERTY CXX_STANDARD 17)
set_property(TARGET liblinear_static PROPERTY CXX_STANDARD 17)

# Handle platform-specific compilation
if(WIN32)
    target_compile_definitions(libsvm_static PRIVATE -D_CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(liblinear_static PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

# Export the source directories for use in main project
set(LIBSVM_INCLUDE_DIR ${libsvm_SOURCE_DIR} PARENT_SCOPE)
set(LIBLINEAR_INCLUDE_DIR ${liblinear_SOURCE_DIR} PARENT_SCOPE)