#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <utility>
#include <stdint.h>

#include "matrix.hpp"

// Btw, I make templates but I'll onlu use f32 lol, in case I'll reuse the code

/*** ACTIVATIONS ***/
/*** ACTIVATIONS ***/
/*** ACTIVATIONS ***/

template<typename T>
Matrix<T> &ReLU(Matrix<T> &input, Matrix<T> &output) {
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return output;
}

template<typename T>
Matrix<T> &Sigmoid(Matrix<T> &input, Matrix<T> &output) {
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = 1 / (1 + exp(-input[i]));
    }
    return output;
}

/**
 * @brief !!!
 * In here we make Quick derivative assuming the input is already the sigmoid output
 * if not, call it as follows: Sigmoid_derivative_quick(Sigmoid(input, output), output)
 *
 */
template<typename T>
Matrix<T> &Sigmoid_derivative_quick(Matrix<T> &input, Matrix<T> &output) {
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] * (1 - input[i]);
    }
    return output;
}

template<typename T>
Matrix<T> &Softmax(Matrix<T> &input, Matrix<T> &output) {
    float sum = 0;
    size_t i = 0;
    for (; i < input.size(); i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }
    for (i = 0; i < input.size(); i++) {
        output[i] /= sum;
    }
    return output;
}

/**
 * @brief
 * Softmax derivative
 * @param input: Data that was passed to the softmax function
 * @param output: The output matrix
 * @return: The derivative of the softmax function
 */
template<typename T>
Matrix<T> Softmax_derivative(Matrix<T> &softmax) {
    return softmax.as_diagonal() - (softmax * softmax.as_transposed());
}

/*** LOSS ***/
/*** LOSS ***/
/*** LOSS ***/

/**
 *@brief
 * Extremelly easy concept with a stupid name that makes you think it's not just a logloss lmao
 * So, it's a log-loss function, but it's called cross-entropy somehow1
 */
template<typename T>
inline double CrossEntropy(const Matrix<T> &predict, unsigned int true_label_index) {
    if (true_label_index >= predict.size()) {
        throw std::invalid_argument("CrossEntropy: Input and target labels must have the same size");
    }
    return -log(predict[true_label_index]);
}

template<typename T>
void MSE(const Matrix<T> &input) {
    // Mean squared error loss function
    // ...
    throw std::runtime_error("MSE not implemented");
}

/*** MULTI-LAYER PERCEPTRON ***/

/**@brief
 * Multi-layer perceptron class workflow:
 *  - Add layers as you want, at any step
 */
template<typename T>
class Mlp {
public:

    enum e_activation_method {
        RELU,
        SIGMOID,
        SOFTMAX
    };

private:

    struct Layer {

        uint32_t    width;
        bool        gradient_cached; // If true, we'll keep data for backprop
        Matrix<T>   weights; // 2D matrix for weights of each link from previous layer (or input) to current layer
        Matrix<T>   biases;  // 1D matrix for biases of each neuron in the current layer
        Matrix<T>   cached_output;  // For back-prop
        enum e_activation_method activation_method;
    };

    uint32_t    _input_size;
    uint32_t    _output_size;
    float       _learning_rate;
    std::vector<struct Layer> _linear_layers;

public:

    /**
     * @brief
     * Multi-layer perceptron class:
     * :param input_size: Number of input features.
     * :param output_size: Number of output features.
     * :param hidden_layers: List of pair, with
     *  the first element being the number of neurons in the layer
     *  and the second element being the activation method.
     * Tip: We use AVX_256 for matrix operations, it would be smart to have
     * the layers width be a multiple of 256/sizof(float) a.k.a 8
     */
    Mlp(uint32_t input_size, uint32_t output_size, float lr,
        const std::vector<std::pair<uint32_t, uint32_t >> &hidden_layers
    ) : _input_size(input_size), _output_size(output_size), _learning_rate(lr) {
        for (auto &layer : hidden_layers) {
            struct Layer l;
            l.width = layer.first;
            l.activation_method = (enum e_activation_method)layer.second;
            l.weights = Matrix<T>(l.width, _input_size).xavier_init();
            l.biases = Matrix<T>(l.width, 1).xavier_init();
            _linear_layers.push_back(l);
            _input_size = l.width;
        }
        _input_size = input_size;
    }

    /**
     * @brief Forward pass:
     * - The forward pass is the process of moving the input data through the network
     * - We return a raw pointer to the output data to avoid allocating a new matrix object
     *   it is however recomanded to either just free it manually or to wrap it back in a matrix object
     *   for automatic memory management
     * @param input: The input data to the network
     * @param gradient_cache: If true, we'll keep data for backprop
     * @return A a raw pointer to the output data
     *
     */
    Matrix<T> forward(const Matrix<T> &input, bool gradient_cache=false) {
        if (_linear_layers.size() == 0) {
            throw std::invalid_argument("Forward error: No layers in the model");
        }
        if (input.rows() != _linear_layers[0].weights.cols()) {
            throw std::invalid_argument("Forward error: Input size must match the first layer width");
        }
        Matrix<T> m = Matrix<T>(input);
        int layer_index = 0; (void)layer_index; // Shut up compiler warning !
        for (auto &layer : _linear_layers) {
            // std::cout << "Layer " << layer_index++ << " :" << std::endl;
            // if (layer_index++ == 2)
            //     std::cout << layer.weights << std::endl;

            m = layer.weights * m; // M is a vector so it has to be on the right side, and multadd don't do that lmao
            m += layer.biases;
            switch (layer.activation_method) {
                case RELU:
                    ReLU(m, m);
                    break;
                case SIGMOID:
                    Sigmoid(m, m);
                    break;
                case SOFTMAX:
                    Softmax(m, m);
                    break;
                default:
                    throw std::invalid_argument("Forward error: Unknown activation method");
            }
            if (gradient_cache) {
                layer.gradient_cached = true;
                layer.cached_output = m;
            }
        }
        return m;
    }

    /**
     * @brief Backward pass:
     * So basicaly: I have no idea what I'm doing
     * But, the spirit is there ! *glitters*
     */
    void backward(const Matrix<T> &nn_result, const Matrix<T> &nn_input, uint true_label_index) {
        // nn_result == _linear_layers.back().cached_output.data, btw
        (void)nn_input;
        uint L = _linear_layers.size() - 1;
        // std::cout << "Backpropagation:" << std::endl;

        // Initialize the loss and loss gradient
        Matrix<T> loss_gradient = Matrix<T>(nn_result);
        loss_gradient[true_label_index] -= 1;
        // loss_gradient = dC/dz[L] = a[L] - y
        // loss_gradient.transpose_inplace(); // Also not sure: [r n, c 1] -> [r 1, c n]
        for (int l = L; l >= 0; l--) {
            // std::cout << "Layer " << l << " :" << std::endl;
            if (!_linear_layers[l].gradient_cached) {
                throw std::invalid_argument("Backward error: No gradient cache for layer,"\
                    "forward pass must be called with gradient_cache=true in order to backpropagate");
            }
            Matrix<T> &a = _linear_layers[l].cached_output;
            Matrix<T> &w = _linear_layers[l].weights;
            Matrix<T> &b = _linear_layers[l].biases;

            // z[l] = w[l] * a[l-1] + b[l]  --> z'[l] = a[l-1] | dz/dw = A^T | dz/db = 1
            Matrix<T> dz_dw_transposed = Matrix<T>(l == 0 ? nn_input : _linear_layers[l-1].cached_output).as_transposed(); // dz/dw[l] = a[l-1]^T

            // da_dz[l] = σ′(z[l])
            Matrix<T> da_dz = Matrix<T>(a.rows(), a.cols());
            switch (_linear_layers[l].activation_method) {
                case RELU: // ReLU will come later
                    // da_dz = ReLU_derivative_quick(a, da_dz);
                    break;
                case SIGMOID:
                    Sigmoid_derivative_quick(a, da_dz);
                    break;
                case SOFTMAX:
                    da_dz.fill(1.0); // Cuz double derivative of softmax is 1 so we shortcut it here
                    break;
                default:
                    throw std::invalid_argument("Backward error: Unknown activation method");
            }
            // local_grad is just da/dz scaled by the propagated loss of the front layer
            // dC_db = da_dz * dz/db = da_dz * 1 = da_dz
            // dC[l]/dw[l] = da/dz * dz/dw^T
            Matrix<T> local_grad = Matrix<T>::hadamard(da_dz, loss_gradient); // Hadamard product
            Matrix<T> dC_dw = local_grad * dz_dw_transposed;
            Matrix<T> dC_db = local_grad;
            dC_dw *= _learning_rate;
            dC_db *= _learning_rate;
            w -= dC_dw;
            b -= dC_db;
            if (l == 0)
                break; // Don't compute a new loss_gradient if we're at the input layer
            loss_gradient =  w.as_transposed() * local_grad; // bet the compiler will tell me that the transoposed is a const r-value and I can't do this
        }
    }

    void train(uint epochs, const std::vector<Matrix<T>>& inputs, const std::vector<uint>& labels) {
        if (inputs.size() != labels.size()) {
            throw std::invalid_argument("Train error: Inputs and labels must have the same size");
        }
        for (uint i = 0; i < epochs; i++) {
            for (uint j = 0; j < inputs.size(); j++) {
                forward(inputs[j], true);
                backward(_linear_layers.back().cached_output, inputs[j], labels[j]);
            }
        }
    }

    inline Matrix<T> predict(const Matrix<T> &input) {
        return forward(input, false);
    }
};

/*

Neurons be like: z=σ(Σi->K a[i]*w[i]+b), but K be changing all the time

σ′(s)=σ(s)⋅(1−σ(s)). Obviously e'=e woaah !! *glitters* (only for sigmoid)
S(a) = Σi->K a[i]*w[i]+b
∂z/∂s=σ′(s)

Calculus for babies:
    The partial derivative of a Matrix*Vector product:
    ∂(M*V)/∂V = V^T, Somehow; Cuz some dude in the XVIth century said we "today we'll transpose lads"
    Thus: in BackProp, The derivative of {dz/dw[l] = a[l-1]^T}
*/