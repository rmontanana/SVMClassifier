/**
 * @file test_main.cpp
 * @brief Main entry point for Catch2 test suite
 *
 * This file contains global test configuration and setup for the SVM classifier
 * test suite. Catch2 will automatically generate the main() function.
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <torch/torch.h>
#include <iostream>

 /**
  * @brief Global test setup
  */
struct GlobalTestSetup {
    GlobalTestSetup()
    {
        // Set PyTorch to single-threaded for reproducible tests
        torch::set_num_threads(1);

        // Set manual seed for reproducibility
        torch::manual_seed(42);

        // Disable PyTorch warnings for cleaner test output
        torch::globalContext().setQEngine(at::QEngine::FBGEMM);

        std::cout << "SVM Classifier Test Suite" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
        std::cout << "Using " << torch::get_num_threads() << " thread(s)" << std::endl;
        std::cout << std::endl;
    }

    ~GlobalTestSetup()
    {
        std::cout << std::endl;
        std::cout << "Test suite completed." << std::endl;
    }
};

// Global setup instance
static GlobalTestSetup global_setup;