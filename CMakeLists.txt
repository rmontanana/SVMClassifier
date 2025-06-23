cmake_minimum_required(VERSION 3.15)
project(SVMClassifier 
    VERSION 1.0.0 
    LANGUAGES CXX
    DESCRIPTION "A C++ library for Support Vector Machine classification using PyTorch"
    HOMEPAGE_URL "https://gitea.rmontanana.es/rmontanana/SVMClassifier"
)
set(PROJECT_AUTHOR "Ricardo Montañana Gómez")
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

# Set policy for FetchContent
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

# Fetch nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Fetch Catch2 for testing
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Add external libraries
add_subdirectory(external)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/libsvm)
include_directories(${CMAKE_SOURCE_DIR}/external/liblinear)

# Create the main library
set(SOURCES
    src/svm_classifier.cpp
    src/data_converter.cpp
    src/multiclass_strategy.cpp
    src/kernel_parameters.cpp
)

set(HEADERS
    include/svm_classifier/svm_classifier.hpp
    include/svm_classifier/data_converter.hpp
    include/svm_classifier/multiclass_strategy.hpp
    include/svm_classifier/kernel_parameters.hpp
    include/svm_classifier/types.hpp
)

# Create library
add_library(svm_classifier STATIC ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(svm_classifier 
    PUBLIC 
        ${TORCH_LIBRARIES}
        nlohmann_json::nlohmann_json
    PRIVATE
        libsvm_static
        liblinear_static
)

# Set include directories
target_include_directories(svm_classifier
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libsvm
        ${CMAKE_CURRENT_SOURCE_DIR}/external/liblinear
)

# Compiler-specific options
target_compile_features(svm_classifier PUBLIC cxx_std_17)

# Set torch CXX flags
set_property(TARGET svm_classifier PROPERTY CXX_STANDARD 17)

# Add examples
add_subdirectory(examples)

# Set default installation paths
include(GNUInstallDirs)
set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME})

# Project information for documentation
set(PROJECT_DESCRIPTION "High-performance Support Vector Machine classifier with scikit-learn compatible API")
set(PROJECT_HOMEPAGE_URL "https://github.com/your-username/svm-classifier")
set(PROJECT_AUTHOR "SVM Classifier Development Team")

# Documentation target
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

if(BUILD_DOCUMENTATION OR DOXYGEN_FOUND)
    find_package(Doxygen QUIET)
    
    if(DOXYGEN_FOUND)
        # Set documentation variables
        set(DOXYGEN_INPUT_DIR "${CMAKE_SOURCE_DIR}")
        set(DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs")
        set(DOXYGEN_INDEX_FILE "${DOXYGEN_OUTPUT_DIR}/html/index.html")
        
        # Check for Graphviz/dot for diagrams
        if(DOXYGEN_DOT_FOUND)
            set(DOXYGEN_DOT_FOUND "YES")
            get_filename_component(DOXYGEN_DOT_PATH ${DOXYGEN_DOT_EXECUTABLE} DIRECTORY)
        else()
            set(DOXYGEN_DOT_FOUND "NO")
            set(DOXYGEN_DOT_PATH "")
        endif()
        
        # Configure the Doxyfile
        configure_file(
            "${CMAKE_SOURCE_DIR}/docs/Doxyfile.in"
            "${CMAKE_BINARY_DIR}/Doxyfile"
            @ONLY
        )
        
        # Create output directory
        file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
        
        # Add custom target for documentation
        add_custom_target(doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_BINARY_DIR}/Doxyfile"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        # Add custom target alias for convenience
        add_custom_target(docs DEPENDS doxygen)
        
        # Install documentation
        if(BUILD_DOCUMENTATION)
            install(
                DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html
                DESTINATION ${CMAKE_INSTALL_DOCDIR}
                COMPONENT documentation
                OPTIONAL
            )
        endif()
        
        message(STATUS "Doxygen found: documentation target 'doxygen' available")
        if(DOXYGEN_DOT_FOUND)
            message(STATUS "Graphviz dot found: enhanced diagrams will be generated")
        else()
            message(STATUS "Graphviz dot not found: basic diagrams only")
        endif()
    else()
        message(WARNING "Doxygen not found: documentation target not available")
    endif()
endif()

# Enable testing
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS svm_classifier
    EXPORT SVMClassifierTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT SVMClassifierTargets
    FILE SVMClassifierTargets.cmake
    NAMESPACE SVMClassifier::
    DESTINATION lib/cmake/SVMClassifier
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    SVMClassifierConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SVMClassifierConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfig.cmake
    INSTALL_DESTINATION lib/cmake/SVMClassifier
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfigVersion.cmake
    DESTINATION lib/cmake/SVMClassifier
)