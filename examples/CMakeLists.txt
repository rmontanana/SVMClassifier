# Examples CMakeLists.txt

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE svm_classifier)
target_include_directories(basic_usage PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(basic_usage PRIVATE cxx_std_17)

# Advanced usage examples (can be added later)
# add_executable(multiclass_example multiclass_example.cpp)
# target_link_libraries(multiclass_example PRIVATE svm_classifier)

# add_executable(hyperparameter_tuning hyperparameter_tuning.cpp)
# target_link_libraries(hyperparameter_tuning PRIVATE svm_classifier)

# add_executable(cross_validation_example cross_validation_example.cpp)
# target_link_libraries(cross_validation_example PRIVATE svm_classifier)

# Installation of examples (optional)
install(TARGETS basic_usage 
    RUNTIME DESTINATION bin/examples
)