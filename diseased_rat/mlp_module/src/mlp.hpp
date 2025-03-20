#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cmath>     // exp, log
#include <utility>   // for std::pair
#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937
#include <iterator>

#include <stdint.h>  // for typedefs
#include "matrix.hpp"

#define EPSILON 1e-8

// Btw, I make templates but I'll onlu use f32 lol, in case I'll reuse the code

/*** MATRIX UTILS ***/

template<typename T>
double matrix_mean(const Matrix<T> &matrix) {
    double sum = 0;
    for (size_t i = 0; i < matrix.size(); i++) {
        sum += matrix[i];
    }
    return sum / matrix.size();
}

template<typename T>
double matrix_variance(const Matrix<T> &matrix, double mean) {
    double sum = 0;
    for (size_t i = 0; i < matrix.size(); i++) {
        sum += pow(matrix[i] - mean, 2);
    }
    return sum / matrix.size();
}

template<typename T>
Matrix<T> &matrix_standardize(Matrix<T> &matrix, const Matrix<T> &gamma, const Matrix<T> &beta) {
    if (matrix.size() != gamma.size() || matrix.size() != beta.size()) {
        throw std::invalid_argument("Matrix standardize error: Gamma and beta size mismatch");
    }
    double mean = matrix_mean(matrix);
    double variance = matrix_variance(matrix, mean);
    double expr = sqrt(variance + EPSILON);
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] = (matrix[i] - mean) / expr * gamma[i] + beta[i];
    }
    return matrix;
}


/*** ACTIVATIONS ***/
/*** ACTIVATIONS ***/
/*** ACTIVATIONS ***/


template<typename T>
Matrix<T> &ReLU(Matrix<T> &input, Matrix<T> &output) {
    if (input.size() != output.size()) {
        throw std::invalid_argument("ReLU error: Input and output size mismatch");
    }
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
    return output;
}

template<typename T>
Matrix<T> &ReLU_derivative(Matrix<T> &input, Matrix<T> &output) {
    if (input.size() != output.size()) {
        throw std::invalid_argument("ReLU_derivative error: Input and output size mismatch");
    }
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] > 0 ? 1 : 0;
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
    T maxVal = input.argmax_value();
    size_t i = 0;
    for (; i < input.size(); i++) {
        output[i] = exp(input[i] - maxVal);
        sum += output[i];
    }
    for (i = 0; i < input.size(); i++) {
        output[i] /= sum + EPSILON;
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

/**
 *@brief
 * Extremelly easy concept with a stupid name that makes you think it's not just a logloss lmao
 * So, it's a log-loss function, but it's called cross-entropy somehow1
 */
template<typename T>
inline double CrossEntropy(const Matrix<T> &predict, size_t true_label_index) {
    if (true_label_index >= predict.size()) {
        throw std::invalid_argument("CrossEntropy: Input and target labels must have the same size");
    }
    auto y = true_label_index == predict.argmax_index() ? 1 : 0;
    auto p = std::clamp((double)predict[true_label_index], EPSILON, 1.0 - EPSILON);
    return -(y*log(p) + (1-y)*log(1-p));
}

template<typename T>
void MSE(const Matrix<T> &input) {
    // Mean squared error loss function
    // ...
    throw std::runtime_error("MSE not implemented");
}

template<typename T>
std::vector<std::pair<Matrix<T>, size_t>>
load_file(const std::string &filename, size_t line_size) {
    std::cout << "Loading file: " << filename << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::invalid_argument("Train from file error: Could not open file");
    }
    std::string line;
    std::vector<std::pair<Matrix<T>, size_t>> data;
    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(line);
        while (std::getline(tokenStream, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 3) {
            throw std::invalid_argument("Train from file error: Invalid file format");
        } if (tokens.size() - 2 != line_size) {
            std::cerr << "Input size: " << line_size << " Data size: " << tokens.size() - 2 << std::endl;
            throw std::invalid_argument("Train from file error: Input size mismatch");
        }
        Matrix<T> input(tokens.size() - 2, 1);
        for (size_t i = 2; i < tokens.size(); i++) {
            input.set_at_flat(i - 2, std::stof(tokens[i]));
        }
        data.emplace_back(input, std::stoi(tokens[1]));
    }
    file.close();
    std::cout << "Succesfully loaded " << data.size() << " samples" << std::endl;
    return data;
}

/*** MULTI-LAYER PERCEPTRON ***/
/*** MULTI-LAYER PERCEPTRON ***/
/*** MULTI-LAYER PERCEPTRON ***/

/**
 * @brief
 * An itterator that will return the indices of the dataset in a shuffled order
 * it stills ensures that all the data will be used
 */
template <typename T>
class ShuffledIterator {
public:
    using DataType = std::vector<std::pair<Matrix<T>, size_t>>; // Features + Label

    ShuffledIterator(const DataType& dataset)
        : dataset_(dataset), indices_(dataset.size()), rng_(std::random_device{}()) {
        std::iota(indices_.begin(), indices_.end(), 0); // Fill with 0,1,2,...,N-1
        shuffle();  // Shuffle indices at start
    }

    void shuffle() {
        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }

    auto begin() { return indices_.begin(); }
    auto end() { return indices_.end(); }
    const auto& operator[](size_t i) const { return dataset_[indices_[i]]; }

private:
    const DataType& dataset_;
    std::vector<size_t> indices_;
    std::mt19937 rng_;
};

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

    typedef struct s_training_report {
        size_t epochs;
        std::vector<double> test_loss;
        std::vector<double> test_accuracy;
        std::vector<double> train_loss;
        std::vector<double> train_accuracy;
    }t_training_report;

private:

    struct Layer {
        uint32_t    width;
        bool        gradient_cached; // If true, we'll keep data for backprop
        Matrix<T>   weights; // 2D matrix for weights of each link from previous layer (or input) to current layer
        Matrix<T>   biases;  // 1D matrix for biases of each neuron in the current layer
        Matrix<T>   gamma;
        Matrix<T>   beta;
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
        const std::vector<std::pair<uint32_t, uint32_t >> &linear_layers
    ) : _input_size(input_size), _output_size(output_size), _learning_rate(lr) {
        for (auto &layer : linear_layers) {
            struct Layer l;
            l.width = layer.first;
            l.activation_method = (enum e_activation_method)layer.second;
            double n;
            if (l.activation_method == SOFTMAX) {
                n = 1.0; // Softmax is a special snowflake
            } else if (l.activation_method == SIGMOID) {
                n = 6.0; // Sigmoid is a bit special too
            } else {
                n = 2.0;
            }
            l.weights = Matrix<T>(l.width, _input_size).xavier_init(n);
            l.biases = Matrix<T>(l.width, 1).xavier_init(n);
            l.gamma = Matrix<T>(l.width, 1).fill(1.0);
            l.beta = Matrix<T>(l.width, 1).fill(0.0);
            _linear_layers.push_back(l);
            _input_size = l.width; // Update last, will be used for next layer
        }
        _input_size = input_size;
        std::cout << "Model created with " << _linear_layers.size() << " layers" << std::endl;
    }

    Mlp(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::invalid_argument("Load model error: Could not open file");
        }
        auto get_matrix_line = [&file]() -> Matrix<T> {
            std::string str;
            if (!std::getline(file, str))
                throw std::invalid_argument("Load model error: Invalid file format");
            return Matrix<T>(str);
        };
        file >> _input_size >> _output_size >> _learning_rate;
        size_t input_size = _input_size;
        int layer_count = 1;
        while (!file.eof() && !file.fail()) {
            std::cerr << "Parsing layer: " << layer_count++ << std::endl;
            _linear_layers.emplace_back(Layer());
            auto &l = _linear_layers.back();
            int32_t tmp;
            if (file >> tmp) {
                l.width = tmp;
            } else throw std::invalid_argument("Invalid file format");
            if (file >> tmp) {
                l.activation_method = (enum e_activation_method)tmp;
            } else throw std::invalid_argument("Invalid file format");
            if (!(file >> std::ws)) throw std::invalid_argument("Invalid file format");// Skip the newline
            l.weights = get_matrix_line();
            l.biases = get_matrix_line();
            l.gamma = get_matrix_line();
            l.beta = get_matrix_line();
            if (l.weights.rows() != l.width || l.weights.cols() != input_size) {
                throw std::invalid_argument("Load model error: weights size mismatch");
            }
            if (l.biases.rows() != l.width || l.biases.cols() != 1)
                throw std::invalid_argument("Load model error: biases size mismatch");
            if (l.gamma.rows() != l.width || l.gamma.cols() != 1)
                throw std::invalid_argument("Load model error: gamma size mismatch");
            if (l.beta.rows() != l.width || l.beta.cols() != 1)
                throw std::invalid_argument("Load model error: beta size mismatch");
            input_size = l.width;
            // Skip empty lines
            while (file.peek() == '\n') {
                file.ignore(1);
            }
        }
        file.close();
        if (_linear_layers.size() == 0) {
            throw std::invalid_argument("Load model error: No layers in the model");
        }
        if (_linear_layers.back().width != _output_size) {
            throw std::invalid_argument("Load model error: Output size mismatch");
        }
        std::cout << "Model created with " << _linear_layers.size() << " layers" << std::endl;
    }

    void save_model(const std::string &filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::invalid_argument("Save model error: Could not open file");
        }
        file << _input_size << " " << _output_size << " " << _learning_rate << std::endl;
        for (auto &layer : _linear_layers) {
            file << layer.width << " " << layer.activation_method << std::endl;
            file << layer.weights.export_string() << std::endl;
            file << layer.biases.export_string() << std::endl;
            file << layer.gamma.export_string() << std::endl;
            file << layer.beta.export_string() << std::endl;
        }
        file.close();
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
        for (auto &layer : _linear_layers) {
            m = layer.weights * m;
            m += layer.biases;
            switch (layer.activation_method) {
                case RELU:
                    // matrix_standardize<T>(m, layer.gamma, layer.beta);
                    ReLU<T>(m, m);
                    break;
                case SIGMOID:
                    // matrix_standardize<T>(m, layer.gamma, layer.beta);
                    Sigmoid<T>(m, m);
                    break;
                case SOFTMAX:
                    Softmax<T>(m, m);
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
     */
    void backward(const Matrix<T> &nn_result, const Matrix<T> &nn_input, size_t true_label_index) {
        // z[l] = w[l] * a[l-1] + b[l]  --> z'[l] = a[l-1] | dz/dw = A^T | dz/db = 1 ; da_dz[l] = σ′(z[l])
        // Initialize the loss and loss gradient
        // loss_gradient = dC/dz[L] = a[L] - y
        Matrix<T> loss_gradient = Matrix<T>(nn_result);
        loss_gradient[true_label_index] -= 1; // only works cuz we always have a softmax at the end, otherwise it's chaos
        for (int l = _linear_layers.size()-1; l >= 0; l--) {
            if (!_linear_layers[l].gradient_cached) {
                throw std::invalid_argument("Backward error: No gradient cache for layer,"\
                    "forward pass must be called with gradient_cache=true in order to backpropagate");
            }

            Matrix<T> &a = _linear_layers[l].cached_output;
            const Matrix<T> &a_prev = l == 0 ? nn_input : _linear_layers[l-1].cached_output;
            Matrix<T> da_dz = Matrix<T>(a.rows(), a.cols());
            switch (_linear_layers[l].activation_method) {
                case RELU: // ReLU will come later
                    ReLU_derivative<T>(a, da_dz);
                    break;
                case SIGMOID:
                    Sigmoid_derivative_quick<T>(a, da_dz);
                    break;
                case SOFTMAX:
                    da_dz.fill(1.0); // Cuz double derivative of softmax is 1 so we shortcut it here
                    // Otherwise, we would have to calculate the derivative of the softmax function
                    // which involve a Jacobian matrix, expensive to compute
                    break;
                default:
                    throw std::invalid_argument("Backward error: Unknown activation method");
            }
            // local_grad is just da/dz scaled by the propagated loss of the front layer
            // dC_db = da_dz * dz/db = da_dz * 1 = da_dz
            // dC[l]/dw[l] = da/dz * dz/dw^T
            Matrix<T> dz_dw_transposed = Matrix<T>(a_prev).as_transposed(); // dz/dw[l] = a[l-1]^T
            Matrix<T> local_grad = Matrix<T>::hadamard(da_dz, loss_gradient); // Hadamard product
            if (l != 0) {
                loss_gradient = _linear_layers[l].weights.as_transposed() * local_grad;
            }
            Matrix<T> dC_dw = local_grad * dz_dw_transposed;
            Matrix<T> dC_db = local_grad;


            dC_dw *= _learning_rate;
            dC_db *= _learning_rate;
            _linear_layers[l].weights -= dC_dw;
            _linear_layers[l].biases-= dC_db;

            // Makes it worse, just leaving this in comment for referencing
            // if (_linear_layers[l].activation_method != SOFTMAX) {
            //     Matrix<T> standardized = Matrix<T>(a);
            //     matrix_standardize<T>(standardized, _linear_layers[l].gamma, _linear_layers[l].beta);
            //     Matrix<T> dC_dgamma = Matrix<T>::hadamard(local_grad, standardized);
            //     Matrix<T> dC_dbeta = local_grad;
            //     dC_dgamma *= _learning_rate;
            //     dC_dbeta *= _learning_rate;
            //     _linear_layers[l].gamma -= dC_dgamma;
            //     _linear_layers[l].beta -= dC_dbeta;
            // }
        }
    }

    /**
     * @brief
     * @param data: A vector of pairs, with the first element being the input data
     * and the second element being the true label index (that shall be activated)
     * @param track_loss: If true, the loss and accuracy will be tracked
     * @return: A struct containing the training report
     */
    void train(size_t epochs, const std::vector<std::pair<Matrix<T>, size_t>> &data) {
        std::cout << "Training with " << data.size() << " samples" << std::endl;
        auto shuffled_it = ShuffledIterator<T>(data);
        double loss;
        for (size_t i = 0; i < epochs; i++) {
            loss = 0;
            for (auto it = shuffled_it.begin(); it != shuffled_it.end(); it++) {
                forward(data[*it].first, true); // ignore return value as it is cached
                backward(_linear_layers.back().cached_output, data[*it].first, data[*it].second);
                loss += CrossEntropy(_linear_layers.back().cached_output, data[*it].second);
            }
            std::cout << "Epoch " << i+1 << "/ " << epochs << " : average loss:" << loss/data.size() << " " << std::endl;
            shuffled_it.shuffle();
        }
    }

    /**
     * @brief
     * Loads the file and sends the data to the train function
     * Parsing policy: A csv file; col 0 id, col 1 label, col 2+ features
     */
    void train_from_file(size_t epochs, const std::string &filename) {
        auto data = load_file<T>(filename, (size_t)_input_size);
        train(epochs, data);
    }

    /**
     * @brief
     * @param filename: The file to test the model with
     */
    void test_from_file(const std::string &filename) {
        auto data = load_file<T>(filename, (size_t)_input_size);
        int hits = 0;
        int i = 1;
        double loss_sum = 0;
        for (auto &d : data) {
            auto result = forward(d.first, false);
            double loss = CrossEntropy(result, d.second);
            if (result.argmax_index() == d.second) {
                hits += 1;
            } else {
                std::cout << "Missed with loss: " << loss << " at row " << i << std::endl;
            }
            loss_sum += loss;
            i++;
        }
        std::cout << "Accuracy: " << hits << "/" << data.size() << " " << (double)hits / data.size()*100 << " %\taverage loss: " << loss_sum / data.size() << std::endl;
    }

    inline Matrix<T> predict(const Matrix<T> &input) {
        return forward(input, false);
    }

    t_training_report train_test_earlystop(const std::string &train_file, const std::string &test_file, size_t epochs) {
        auto train_data = load_file<T>(train_file, (size_t)_input_size);
        auto test_data = load_file<T>(test_file, (size_t)_input_size);
        if (train_data.size() == 0 || test_data.size() == 0) {
            throw std::invalid_argument("Train/Test error: Empty dataset");
        }

        t_training_report training_report;
        training_report.epochs = epochs;
        training_report.train_loss.reserve(epochs);
        training_report.train_accuracy.reserve(epochs);
        training_report.test_accuracy.reserve(epochs);
        training_report.test_loss.reserve(epochs);

        const size_t patience = std::max((size_t)50, std::min((size_t)3000, epochs / 5));
        const double threshold = 0.001;
        double best_loss = 999;
        size_t patience_counter = 0;
        std::vector<struct Layer> best_model;
        int best_model_epoch = 0;

        std::cout << "Training with " << train_data.size() << " samples" << std::endl;
        std::cout << "Testing with " << test_data.size() << " samples" << std::endl;
        std::cout << "Patience: " << patience << std::endl;

        auto shuffled_it = ShuffledIterator<T>(train_data);
        for (size_t i = 0; i < epochs; i++) {

            // Train
            for (auto it = shuffled_it.begin(); it != shuffled_it.end(); it++) {
                forward(train_data[*it].first, true);
                backward(_linear_layers.back().cached_output, train_data[*it].first, train_data[*it].second);
            }
            shuffled_it.shuffle();

            // Test on training set
            double loss = 0;
            int hits = 0;
            for (auto &d : train_data) {
                auto result = forward(d.first, false);
                if (result.argmax_index() == d.second)
                    hits += 1;
                loss += CrossEntropy(result, d.second);
            }
            training_report.train_loss.push_back(loss / train_data.size());
            training_report.train_accuracy.push_back((double)hits / train_data.size());
            // Test on validation set
            loss = 0;
            hits = 0;
            for (auto &d : test_data) {
                auto result = forward(d.first, false);
                if (result.argmax_index() == d.second)
                    hits += 1;
                loss += CrossEntropy(result, d.second);
            }
            double accuracy = (double)hits / test_data.size();
            double loss_avg = loss / test_data.size();
            training_report.test_loss.push_back(loss_avg);
            training_report.test_accuracy.push_back(accuracy);
            std::cout << "Epoch " << i+1 << "/ " << epochs << " : loss average: " << loss_avg << "  Accuracy: " << accuracy << std::endl;
            // Caching the model
            if (loss_avg < best_loss) {
                best_loss = loss_avg;
                best_model = _linear_layers;
                best_model_epoch = i;
                if (best_loss - loss_avg > threshold)
                    patience_counter = 0;
            } else {
                patience_counter += 1;
                if (patience_counter > patience) {
                    // The best model hasn't improved for a while, let's stop here
                    std::cout << "Early stopping at epoch " << i+1 << " using model from epoch " << best_model_epoch+1 << std::endl;
                    _linear_layers = best_model;
                    break;
                }
            }
        }
        _linear_layers = best_model; // Restore the best model
        double loss = 0;
        int hits = 0;
        for (auto &d : test_data) {
            auto result = forward(d.first, false);
            if (result.argmax_index() == d.second)
                hits += 1;
            loss += CrossEntropy(result, d.second);
        }
        std::cout << "Best model : loss: " << loss/test_data.size() << "  Accuracy: " << (double)hits / test_data.size() << std::endl;
        return training_report;
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
