# CPack configuration for SVMClassifier

set(CPACK_PACKAGE_NAME "SVMClassifier")
set(CPACK_PACKAGE_VENDOR "SVMClassifier Development Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance SVM classifier with scikit-learn compatible API")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# Package description
set(CPACK_PACKAGE_DESCRIPTION "
SVMClassifier is a high-performance Support Vector Machine classifier 
implementation in C++ with a scikit-learn compatible API. It provides:

- Multiple kernel support (linear, RBF, polynomial, sigmoid)
- Multiclass classification (One-vs-Rest and One-vs-One)
- PyTorch tensor integration
- JSON configuration
- Comprehensive testing suite
- Cross-validation and grid search capabilities

The library automatically selects between liblinear (for linear kernels) 
and libsvm (for non-linear kernels) to ensure optimal performance.
")

# Contact information
set(CPACK_PACKAGE_CONTACT "svm-classifier@example.com")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/your-username/svm-classifier")

# License
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Installation directories
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

#-----------------------------------------------------------------------------
# Platform-specific settings
#-----------------------------------------------------------------------------

if(WIN32)
    # Windows-specific settings
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "SVM Classifier C++")
    set(CPACK_NSIS_PACKAGE_NAME "SVMClassifier")
    set(CPACK_NSIS_HELP_LINK "https://github.com/your-username/svm-classifier")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/your-username/svm-classifier")
    set(CPACK_NSIS_CONTACT "svm-classifier@example.com")
    set(CPACK_NSIS_MODIFY_PATH ON)
    
    # Add PyTorch requirement note
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
        MessageBox MB_OK 'Please ensure PyTorch C++ (libtorch) is installed and accessible via PATH or CMAKE_PREFIX_PATH.'
    ")
    
elseif(APPLE)
    # macOS-specific settings
    set(CPACK_GENERATOR "TGZ;DragNDrop")
    set(CPACK_DMG_VOLUME_NAME "SVMClassifier")
    set(CPACK_DMG_FORMAT "UDZO")
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_SOURCE_DIR}/packaging/dmg_background.png")
    
else()
    # Linux-specific settings
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
    
    # Debian package settings
    set(CPACK_DEBIAN_PACKAGE_SECTION "science")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libblas3, liblapack3")
    set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libtorch-dev")
    set(CPACK_DEBIAN_PACKAGE_SUGGESTS "cmake, build-essential")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/your-username/svm-classifier")
    
    # RPM package settings
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libstdc++, blas, lapack")
    set(CPACK_RPM_PACKAGE_SUGGESTS "cmake, gcc-c++, libtorch-devel")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/your-username/svm-classifier")
    
    # Set package file names
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
endif()

#-----------------------------------------------------------------------------
# Component-based packaging
#-----------------------------------------------------------------------------

# Runtime component (libraries)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Runtime Libraries")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "SVMClassifier runtime libraries")
set(CPACK_COMPONENT_RUNTIME_REQUIRED TRUE)

# Development component (headers, cmake files)
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development Files")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and CMake configuration files for development")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS runtime)

# Examples component
set(CPACK_COMPONENT_EXAMPLES_DISPLAY_NAME "Examples")
set(CPACK_COMPONENT_EXAMPLES_DESCRIPTION "Example applications demonstrating SVMClassifier usage")
set(CPACK_COMPONENT_EXAMPLES_DEPENDS runtime)

# Documentation component
set(CPACK_COMPONENT_DOCUMENTATION_DISPLAY_NAME "Documentation")
set(CPACK_COMPONENT_DOCUMENTATION_DESCRIPTION "API documentation and user guides")

# Archive settings
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

#-----------------------------------------------------------------------------
# Advanced packaging options
#-----------------------------------------------------------------------------

# Source package
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "/\\.git/"
    "/\\.github/"
    "/build/"
    "/\\.vscode/"
    "/\\.idea/"
    "\\.DS_Store"
    "\\.gitignore"
    "\\.gitmodules"
    ".*~$"
    "\\.swp$"
    "\\.orig$"
    "/CMakeLists\\.txt\\.user$"
    "/Makefile$"
    "/CMakeCache\\.txt$"
    "/CMakeFiles/"
    "/cmake_install\\.cmake$"
    "/install_manifest\\.txt$"
    "/CPackConfig\\.cmake$"
    "/CPackSourceConfig\\.cmake$"
    "/_CPack_Packages/"
    "\\.tar\\.gz$"
    "\\.tar\\.bz2$"
    "\\.tar\\.Z$"
    "\\.svn/"
    "\\.cvsignore$"
    "\\.bzr/"
    "\\.hg/"
    "\\.git/"
    "\\.DS_Store$"
)

#-----------------------------------------------------------------------------
# Testing and validation
#-----------------------------------------------------------------------------

# Add post-install test option
option(CPACK_PACKAGE_INSTALL_TESTS "Include tests in package for post-install validation" OFF)

if(CPACK_PACKAGE_INSTALL_TESTS)
    install(TARGETS svm_classifier_tests
        RUNTIME DESTINATION bin/tests
        COMPONENT testing
    )
    
    set(CPACK_COMPONENT_TESTING_DISPLAY_NAME "Test Suite")
    set(CPACK_COMPONENT_TESTING_DESCRIPTION "Test suite for post-installation validation")
    set(CPACK_COMPONENT_TESTING_DEPENDS runtime development)
endif()

#-----------------------------------------------------------------------------
# Include CPack
#-----------------------------------------------------------------------------

include(CPack)