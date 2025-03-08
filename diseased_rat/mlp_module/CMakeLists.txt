# Boilerplate
cmake_minimum_required(VERSION 3.14)
project(mlp_module)

# Compiler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx -march=native")

# Dependencies and includes
find_package(pybind11 REQUIRED)

set(TARGET_NAME _mlp_module)

# Add the module
pybind11_add_module(${TARGET_NAME} pybind_mlp.cpp src/mlp.cpp)

# Include headers
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/headers)

# Add external libraries
target_link_libraries(${TARGET_NAME} PRIVATE pybind11::module)

# Define the binary type and name
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "" SUFFIX ".so")
