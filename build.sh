#!/bin/bash

# Exit on errors
set -e

# Create virtual environment if it does not exist
if [ ! -d "./.venv" ]; then
    echo "Venv not found, creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Build C++ files
cd diseased_rat

# rm -rf build
if [ ! -d "./build" ]; then
    mkdir build
fi
cd build

cmake .. -Dpybind11_DIR="$(find ../../.venv/lib/ -type d -path '*/pybind11/share/cmake/pybind11')"
cmake --build .

cd ..

# Create symbolic links
ln -sf build/lib/_matrix_module.so ./matrix_module/
ln -sf build/lib/_mlp_module.so ./mlp_module/

# Install compiled library as a Python package
pip install .

cd ..
