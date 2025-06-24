#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>
#include <iostream>

using namespace svm_classifier;

int main() {
    try {
        std::cout << "Creating linear SVM..." << std::endl;
        SVMClassifier svm(KernelType::LINEAR, 1.0);
        
        std::cout << "Generating data..." << std::endl;
        torch::manual_seed(42);
        auto X = torch::randn({50, 3});
        auto y = torch::randint(0, 2, {50});
        
        std::cout << "Training SVM..." << std::endl;
        auto metrics = svm.fit(X, y);
        std::cout << "Training completed successfully!" << std::endl;
        std::cout << "Is fitted: " << svm.is_fitted() << std::endl;
        
        std::cout << "\nChanging to RBF kernel..." << std::endl;
        nlohmann::json new_params = {{"kernel", "rbf"}};
        svm.set_parameters(new_params);
        std::cout << "Parameters changed successfully!" << std::endl;
        std::cout << "Is fitted after param change: " << svm.is_fitted() << std::endl;
        
        std::cout << "\nAll operations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}