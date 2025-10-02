import os
import re
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.files import copy, load


class SVMClassifierConan(ConanFile):
    name = "svmclassifier"
    license = "MIT"
    author = "Ricardo Montañana Gómez"
    url = "https://gitea.rmontanana.es/rmontanana/SVMClassifier"
    description = (
        "A C++ library for Support Vector Machine classification using PyTorch"
    )
    topics = ("svm", "machine-learning", "pytorch", "classification")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "enable_coverage": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "enable_coverage": False,
    }
    exports_sources = (
        "CMakeLists.txt",
        "src/*",
        "include/*",
        "cmake/*",
        "tests/*",
        "examples/*",
        "LICENSE",
        "README.md",
    )

    def set_version(self):
        """Extract version from CMakeLists.txt"""
        cmake_file = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        version_match = re.search(
            r"project\([^)]*VERSION\s+(\d+\.\d+\.\d+)", cmake_file
        )
        if version_match:
            self.version = version_match.group(1)
        else:
            raise ValueError("Could not extract version from CMakeLists.txt")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        self.requires("nlohmann_json/3.11.3")
        self.requires("libtorch/2.7.1")
        # self.requires("libsvm/333")

    def build_requirements(self):
        self.test_requires("catch2/3.4.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_DOCUMENTATION"] = "OFF"
        tc.variables["ENABLE_COVERAGE"] = self.options.enable_coverage
        # Allow users to pass CMAKE_PREFIX_PATH for libtorch
        if "CMAKE_PREFIX_PATH" in os.environ:
            tc.variables["CMAKE_PREFIX_PATH"] = os.environ["CMAKE_PREFIX_PATH"]
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["svm_classifier"]
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.set_property("cmake_file_name", "SVMClassifier")
        self.cpp_info.set_property("cmake_target_name", "SVMClassifier::svm_classifier")

        # CMake config files
        self.cpp_info.builddirs = [
            os.path.join("lib", "cmake", "SVMClassifier"),
        ]
