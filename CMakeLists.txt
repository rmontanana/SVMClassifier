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

# Link libraries with PRIVATE linkage to avoid export issues
target_link_libraries(svm_classifier 
    PUBLIC 
        ${TORCH_LIBRARIES}
    PRIVATE
        nlohmann_json::nlohmann_json
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

# Memory check target (if valgrind is available)
find_program(VALGRIND valgrind)
if(VALGRIND)
    add_custom_target(test_memcheck
        COMMAND ${VALGRIND} --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --error-exitcode=1 $<TARGET_FILE:svm_classifier_tests>
        DEPENDS svm_classifier_tests
        COMMENT "Running memory check with Valgrind"
    )
    message(STATUS "Memory check target 'test_memcheck' available")
endif()

# Performance profiling target (if perf is available)
find_program(PERF perf)
if(PERF)
    add_custom_target(test_profile
        COMMAND ${PERF} record --call-graph=dwarf $<TARGET_FILE:svm_classifier_tests>
        DEPENDS svm_classifier_tests
        COMMENT "Running performance profiling with perf"
    )
    message(STATUS "Performance profiling target 'test_profile' available")
endif()

# Enable testing
enable_testing()
add_subdirectory(tests)

# Add examples
add_subdirectory(examples)

# Installation - Fixed version
install(TARGETS svm_classifier
    EXPORT SVMClassifierTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Create package configuration files
include(CMakePackageConfigHelpers)

# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SVMClassifierConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVMClassifier
)

# Create version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SVMClassifierConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVMClassifier
)

# Install export targets
install(EXPORT SVMClassifierTargets
    FILE SVMClassifierTargets.cmake
    NAMESPACE SVMClassifier::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVMClassifier
)

# Package configuration
set(CPACK_PACKAGE_NAME "SVMClassifier")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance SVM classifier with scikit-learn compatible API")
set(CPACK_PACKAGE_VENDOR "SVMClassifier Development Team")
set(CPACK_RESOURCE_FILE_README ${CMAKE_SOURCE_DIR}/README.md)

# Platform-specific package settings
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libblas3, liblapack3")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libstdc++, blas, lapack")
endif()

include(CPack)