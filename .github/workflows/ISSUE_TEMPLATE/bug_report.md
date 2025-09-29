---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

A clear and concise description of what actually happened.

## 💻 Environment

**System Information:**
- OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 10]
- Compiler: [e.g. GCC 9.4.0, Clang 12.0.0, MSVC 2019]
- CMake Version: [e.g. 3.20.0]
- PyTorch Version: [e.g. 2.8.0]

**Library Versions:**
- SVM Classifier Version: [e.g. 1.0.0]
- libsvm Version: [if known]
- liblinear Version: [if known]

## 📋 Minimal Reproduction Code

```cpp
#include <svm_classifier/svm_classifier.hpp>
#include <torch/torch.h>

int main() {
    // Your minimal code that reproduces the issue
    using namespace svm_classifier;
    
    // Example:
    auto X = torch::randn({100, 4});
    auto y = torch::randint(0, 3, {100});
    
    SVMClassifier svm(KernelType::RBF);
    auto metrics = svm.fit(X, y);  // Error occurs here
    
    return 0;
}
```

**Compilation command:**
```bash
g++ -std=c++17 reproduce_bug.cpp -lsvm_classifier -ltorch -ltorch_cpu -o reproduce_bug
```

## 📊 Error Output

```
Paste the full error message, stack trace, or unexpected output here
```

## 🔍 Additional Context

Add any other context about the problem here:

- Were you following a specific tutorial or documentation?
- Did this work in a previous version?
- Are there any workarounds you've found?
- Any additional error logs or debugging information?

## 📎 Attachments

If applicable, add:
- Screenshots of error messages
- Log files
- Core dumps (if available)
- Example datasets (if relevant and small)

## ✅ Checklist

- [ ] I have searched for existing issues that might be related to this bug
- [ ] I have provided a minimal reproduction case
- [ ] I have included all relevant environment information
- [ ] I have tested this with the latest version of the library
- [ ] I have checked that my build environment meets the requirements