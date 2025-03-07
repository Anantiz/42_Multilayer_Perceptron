cmake_minimum_required(VERSION 3.14)
project(pybind_modules)

# Set the output directory for the libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add subdirectories for each module
add_subdirectory(matrix_module)
add_subdirectory(mlp_module)