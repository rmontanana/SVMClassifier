# Examples CMakeLists.txt

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE svm_classifier
    nlohmann_json::nlohmann_json
)
target_include_directories(basic_usage PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(basic_usage PRIVATE cxx_std_17)

# Advanced usage example
add_executable(advanced_usage advanced_usage.cpp)
target_link_libraries(advanced_usage PRIVATE svm_classifier
    nlohmann_json::nlohmann_json
)
target_include_directories(advanced_usage PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(advanced_usage PRIVATE cxx_std_17)

# Installation of examples (optional)
install(TARGETS basic_usage advanced_usage
    RUNTIME DESTINATION bin/examples
)