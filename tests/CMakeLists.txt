# Tests CMakeLists.txt

# Find Catch2 (should already be available from main CMakeLists.txt)
# find_package(Catch2 3 REQUIRED)

# Include Catch2 extras for automatic test discovery
include(Catch)

# Test sources
set(TEST_SOURCES
    test_main.cpp
    test_svm_classifier.cpp
    test_data_converter.cpp
    test_multiclass_strategy.cpp
    test_kernel_parameters.cpp
)

# Create test executable
add_executable(svm_classifier_tests ${TEST_SOURCES})
add_test(NAME svm_classifier_tests COMMAND svm_classifier_tests)
# Link with the main library and Catch2
target_link_libraries(svm_classifier_tests 
    PRIVATE 
        svm_classifier
        Catch2::Catch2WithMain
        nlohmann_json::nlohmann_json
        $<$<TARGET_EXISTS:blas_functions>:blas_functions>
)

# Set include directories - Handle external libraries dynamically
target_include_directories(svm_classifier_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Add Catch2 include directories explicitly
if(TARGET Catch2::Catch2)
    get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2 INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(svm_classifier_tests PRIVATE ${CATCH2_INCLUDE_DIRS})
endif()

# Also add the source directory of Catch2 if available
if(catch2_SOURCE_DIR)
    target_include_directories(svm_classifier_tests
        PRIVATE
            ${catch2_SOURCE_DIR}/src
    )
endif()

# Add libsvm include directory if available
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../_deps/libsvm-src")
    target_include_directories(svm_classifier_tests 
        PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../_deps/libsvm-src"
    )
endif()

# Add liblinear include directories if available
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/../_deps/liblinear-src")
    target_include_directories(svm_classifier_tests 
        PRIVATE 
            "${CMAKE_CURRENT_BINARY_DIR}/../_deps/liblinear-src"
            "${CMAKE_CURRENT_BINARY_DIR}/../_deps/liblinear-src/blas"
    )
endif()

# Compiler flags for tests
target_compile_features(svm_classifier_tests PRIVATE cxx_std_17)

# Add compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(svm_classifier_tests PRIVATE 
        -Wall -Wextra -pedantic -Wno-unused-parameter
    )
endif()

# Discover tests automatically
catch_discover_tests(svm_classifier_tests)

# Add custom targets for different test categories
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -L "unit" --output-on-failure
    DEPENDS svm_classifier_tests
    COMMENT "Running unit tests"
)

add_custom_target(test_integration
    COMMAND ${CMAKE_CTEST_COMMAND} -L "integration" --output-on-failure
    DEPENDS svm_classifier_tests
    COMMENT "Running integration tests"
)

add_custom_target(test_performance
    COMMAND ${CMAKE_CTEST_COMMAND} -L "performance" --output-on-failure
    DEPENDS svm_classifier_tests
    COMMENT "Running performance tests"
)

add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS svm_classifier_tests
    COMMENT "Running all tests"
)

# Coverage target (if gcov/lcov available)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_program(GCOV_EXECUTABLE gcov)
    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)
    
    if(GCOV_EXECUTABLE AND LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        target_compile_options(svm_classifier_tests PRIVATE --coverage)
        target_link_options(svm_classifier_tests PRIVATE --coverage)
        
        add_custom_target(coverage
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            COMMAND ${LCOV_EXECUTABLE} --capture --directory . --output-file coverage.info
            COMMAND ${LCOV_EXECUTABLE} --remove coverage.info '/usr/*' '*/external/*' '*/tests/*' --output-file coverage_filtered.info
            COMMAND ${GENHTML_EXECUTABLE} coverage_filtered.info --output-directory coverage_html
            DEPENDS svm_classifier_tests
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
        
        message(STATUS "Code coverage target 'coverage' available")
    endif()
endif()

# Add memory check with valgrind if available
find_program(VALGRIND_EXECUTABLE valgrind)
if(VALGRIND_EXECUTABLE)
    add_custom_target(tests_memcheck
        COMMAND ${VALGRIND_EXECUTABLE} --tool=memcheck --leak-check=full --show-leak-kinds=all 
                --track-origins=yes --verbose --error-exitcode=1 
                $<TARGET_FILE:svm_classifier_tests>
        DEPENDS svm_classifier_tests
        COMMENT "Running tests with valgrind memory check"
    )
    
    message(STATUS "Memory check target 'tests_memcheck' available")
endif()

# Performance profiling with perf if available
find_program(PERF_EXECUTABLE perf)
if(PERF_EXECUTABLE)
    add_custom_target(tests_profile
        COMMAND ${PERF_EXECUTABLE} record -g $<TARGET_FILE:svm_classifier_tests> [performance]
        COMMAND ${PERF_EXECUTABLE} report
        DEPENDS svm_classifier_tests
        COMMENT "Running performance tests with profiling"
    )
    
    message(STATUS "Performance profiling target 'tests_profile' available")
endif()

# Set test properties
set_tests_properties(svm_classifier_tests PROPERTIES
    TIMEOUT 300  # 5 minutes timeout
    ENVIRONMENT "TORCH_NUM_THREADS=1"  # Single-threaded for reproducible results
)