/**
 * @file test_main.cpp
 * @brief Main entry point for Catch2 test suite
 *
 * This file contains global test configuration and setup for the SVM classifier
 * test suite. Catch2 will automatically generate the main() function.
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
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

        // Set default quantized engine (safer option for compatibility)
        try {
            // Try FBGEMM first, fallback to default if not available
            torch::globalContext().setQEngine(at::QEngine::FBGEMM);
        }
        catch (const std::exception&) {
            // Use default engine if FBGEMM is not available
            torch::globalContext().setQEngine(at::QEngine::NoQEngine);
        }
    }
};

// Global setup instance
static GlobalTestSetup global_setup;