#!/bin/bash

if [ ! -d "./.venv" ] then
    python3 -m venv .venv
end

source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt

cd diseased_rat # The cpp files
    rm -rf build
    mkdir build
    cd build
        cmake ..
        cmake --build .
    cd ..
        ln -sf build/lib/_matrix_module.so -t ./matrix_module
        ln -sf build/lib/_mlp_module.so -t ./mlp_module
        # install the compiled library as a python package
        pip install .
    cd ..
