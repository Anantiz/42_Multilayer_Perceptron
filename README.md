# Multilayer Perceptron 🧠

A binary classifier MLP implementation from scratch as part of 42school's Math branch.

## Overview

This project implements a multilayer perceptron neural network in C++ with Python bindings for training and visualization. The core architecture is designed for binary classification tasks.

## Features

- **Pure C++ implementation** with two dynamic libraries: `mlp` and `matrix`
- **Python bindings** for easy integration and scripting
- **Training visualization** and data parsing through Python interface
- **SIMD optimization** attempts in the matrix library
- **Flexible architecture** with configurable hidden layers and activation functions

## Limitations

- The C++ library is specifically tailored to this project's requirements
- The Matrix module is basic; see [42_Matrix](https://github.com/Anantiz/42_Matrix) for a more advanced generic template library
- Architecture could be generalized but currently optimized for the subject's dataset

## Requirements

- Python 3
- CMake
- C++ compiler

## Installation

```sh
bash build.sh
```

## Usage

```sh
# Train the model, evaluate it, and generate visualizations
python3 bicolor_snake/train_test.py \
  --training_set_file ./train_set.csv \
  --test_set_file ./test_set.csv \
  --epochs 150 \
  --input_size 30 \
  --output_size 2 \
  --hidden_layers '(30, relu)' '(16, relu)' '(8, sigmoid)' '(2, softmax)' \
  --learning_rate 0.00003
```

## Technical Details

- **Matrix operations** with SIMD optimization attempts
- **Linear layers** with softmax output
- **Gradient descent** with backpropagation
- **CMake build system** and Python packaging

## Skills Learned

- Gradient descent and backpropagation algorithms
- C++ to Python bindings
- CMake project configuration
- Neural network fundamentals
