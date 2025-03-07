#!python3
import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom build_ext command to use CMake
class CMakeBuildExt(build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the project.")

        # Create a build directory
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake
        cmake_command = [
            "cmake",
            "-S", ".",  # Source directory
            "-B", build_dir,  # Build directory
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(os.path.abspath(self.build_lib)),
        ]
        subprocess.check_call(cmake_command)

        # Build the project
        subprocess.check_call(["cmake", "--build", build_dir])

# Define the package
setup(
    name="my_modules",
    version="0.1",
    author="🐜1️⃣",
    description="Python bindings for matrix and MLP modules",
    long_description="",
    ext_modules=[],  # No direct extensions, handled by CMake
    cmdclass={"build_ext": CMakeBuildExt},
    packages=["matrix_module", "mlp_module"],  # Include Python packages
    package_dir={"": "."},  # Look for packages in the root directory
    zip_safe=False,
)
