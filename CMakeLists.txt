cmake_minimum_required(VERSION 3.15)
project(SVMClassifier VERSION 1.0.0 LANGUAGES CXX)

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