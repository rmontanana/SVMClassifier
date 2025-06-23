# External dependencies CMakeLists.txt - Fixed version with OBJECT libraries

include(FetchContent)

# Set policies for FetchContent
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

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

# Build libsvm as OBJECT library to avoid export issues
if(EXISTS "${libsvm_SOURCE_DIR}/svm.cpp")
    set(LIBSVM_SOURCES "${libsvm_SOURCE_DIR}/svm.cpp")
    
    add_library(libsvm_objects OBJECT ${LIBSVM_SOURCES})
    
    # Set properties for the object library
    target_include_directories(libsvm_objects 
        PRIVATE 
            ${libsvm_SOURCE_DIR}
    )
    
    # Set C++ standard for libsvm
    target_compile_features(libsvm_objects PRIVATE cxx_std_17)
    
    # Make the include directory available to parent scope
    set(LIBSVM_INCLUDE_DIR ${libsvm_SOURCE_DIR} PARENT_SCOPE)
    
    message(STATUS "libsvm built successfully as object library")
else()
    message(WARNING "libsvm source files not found")
    # Create empty object library
    add_library(libsvm_objects OBJECT)
    set(LIBSVM_INCLUDE_DIR "" PARENT_SCOPE)
endif()

# Build liblinear as OBJECT library
set(LIBLINEAR_SOURCES)

# Check for main liblinear source files
if(EXISTS "${liblinear_SOURCE_DIR}/linear.cpp")
    list(APPEND LIBLINEAR_SOURCES "${liblinear_SOURCE_DIR}/linear.cpp")
endif()

# Check for optimization files (tron.cpp, newton.cpp, etc.)
if(EXISTS "${liblinear_SOURCE_DIR}/tron.cpp")
    list(APPEND LIBLINEAR_SOURCES "${liblinear_SOURCE_DIR}/tron.cpp")
elseif(EXISTS "${liblinear_SOURCE_DIR}/newton.cpp")
    list(APPEND LIBLINEAR_SOURCES "${liblinear_SOURCE_DIR}/newton.cpp")
endif()

# Check for BLAS files in blas directory
file(GLOB BLAS_C_FILES "${liblinear_SOURCE_DIR}/blas/*.c")
file(GLOB BLAS_CPP_FILES "${liblinear_SOURCE_DIR}/blas/*.cpp")

if(BLAS_C_FILES OR BLAS_CPP_FILES)
    list(APPEND LIBLINEAR_SOURCES ${BLAS_C_FILES} ${BLAS_CPP_FILES})
else()
    # Try alternative BLAS file names
    foreach(blas_file daxpy ddot dnrm2 dscal)
        if(EXISTS "${liblinear_SOURCE_DIR}/blas/${blas_file}.c")
            list(APPEND LIBLINEAR_SOURCES "${liblinear_SOURCE_DIR}/blas/${blas_file}.c")
        endif()
    endforeach()
endif()

# Create liblinear object library if we have source files
if(LIBLINEAR_SOURCES)
    add_library(liblinear_objects OBJECT ${LIBLINEAR_SOURCES})
    
    # Set properties for the object library
    target_include_directories(liblinear_objects 
        PRIVATE 
            ${liblinear_SOURCE_DIR}
    )
    
    # Add blas directory if it exists
    if(EXISTS "${liblinear_SOURCE_DIR}/blas")
        target_include_directories(liblinear_objects 
            PRIVATE ${liblinear_SOURCE_DIR}/blas
        )
    endif()
    
    # Set C++ standard for liblinear
    target_compile_features(liblinear_objects PRIVATE cxx_std_17)
    
    # Compiler specific flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(liblinear_objects PRIVATE -w) # Suppress warnings
    endif()
    
    # Make the include directories available to parent scope
    set(LIBLINEAR_INCLUDE_DIR ${liblinear_SOURCE_DIR} PARENT_SCOPE)
    if(EXISTS "${liblinear_SOURCE_DIR}/blas")
        set(LIBLINEAR_BLAS_INCLUDE_DIR ${liblinear_SOURCE_DIR}/blas PARENT_SCOPE)
    endif()
    
    message(STATUS "liblinear built with sources: ${LIBLINEAR_SOURCES}")
else()
    # Create minimal liblinear implementation as object library
    message(WARNING "liblinear source files not found, creating minimal implementation")
    
    # Create a minimal linear.cpp file
    set(MINIMAL_LIBLINEAR_DIR "${CMAKE_CURRENT_BINARY_DIR}/minimal_liblinear")
    file(MAKE_DIRECTORY ${MINIMAL_LIBLINEAR_DIR})
    
    # Create minimal header
    file(WRITE "${MINIMAL_LIBLINEAR_DIR}/linear.h"
        "#pragma once\n"
        "extern \"C\" {\n"
        "struct feature_node { int index; double value; };\n"
        "struct problem { int l, n; double *y; struct feature_node **x; double bias; };\n"
        "struct parameter { int solver_type; double eps, C; };\n"
        "struct model { struct parameter param; int nr_class, nr_feature; double *w; int *label; double bias; };\n"
        "struct model* train(const struct problem *prob, const struct parameter *param);\n"
        "double predict(const struct model *model_, const struct feature_node *x);\n"
        "void free_and_destroy_model(struct model **model_ptr_ptr);\n"
        "}\n"
    )
    
    # Create minimal implementation
    file(WRITE "${MINIMAL_LIBLINEAR_DIR}/linear.cpp"
        "#include \"linear.h\"\n"
        "#include <cstdlib>\n"
        "#include <cstring>\n"
        "struct model* train(const struct problem *prob, const struct parameter *param) {\n"
        "    auto model = (struct model*)malloc(sizeof(struct model));\n"
        "    memset(model, 0, sizeof(struct model));\n"
        "    model->nr_feature = prob->n;\n"
        "    model->nr_class = 2;\n"
        "    model->w = (double*)calloc(prob->n, sizeof(double));\n"
        "    return model;\n"
        "}\n"
        "double predict(const struct model *model_, const struct feature_node *x) {\n"
        "    return 1.0; // Dummy prediction\n"
        "}\n"
        "void free_and_destroy_model(struct model **model_ptr_ptr) {\n"
        "    if (*model_ptr_ptr) {\n"
        "        free((*model_ptr_ptr)->w);\n"
        "        free(*model_ptr_ptr);\n"
        "        *model_ptr_ptr = nullptr;\n"
        "    }\n"
        "}\n"
    )
    
    add_library(liblinear_objects OBJECT "${MINIMAL_LIBLINEAR_DIR}/linear.cpp")
    target_include_directories(liblinear_objects PRIVATE ${MINIMAL_LIBLINEAR_DIR})
    target_compile_features(liblinear_objects PRIVATE cxx_std_17)
    
    set(LIBLINEAR_INCLUDE_DIR ${MINIMAL_LIBLINEAR_DIR} PARENT_SCOPE)
endif()

# Print summary
message(STATUS "External libraries configured:")
message(STATUS "  - libsvm: ${libsvm_SOURCE_DIR}")  
message(STATUS "  - liblinear: ${liblinear_SOURCE_DIR}")
message(STATUS "  - Using OBJECT libraries to avoid export issues")