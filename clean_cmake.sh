#!/bin/bash

# Define patterns to search for
patterns=(
    "CMakeCache.txt"
    "CMakeFiles"
    "Makefile"
    "cmake_install.cmake"
    "build"
)

# Find and remove each pattern
for pattern in "${patterns[@]}"; do
    find ./diseased_rat -name "$pattern" -exec rm -rf {} +
    echo "Removed $pattern"
done

echo "CMake cleanup complete."
