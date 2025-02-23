#include <stdint.h>

const int DTYPE_SIZE[] = {sizeof(float), \
    sizeof(double), \
    sizeof(long double), \
    sizeof(int8_t), \
    sizeof(int16_t), \
    sizeof(int32_t), \
    sizeof(int64_t), \
    sizeof(uint8_t), \
    sizeof(uint16_t), \
    sizeof(uint32_t), \
    sizeof(uint64_t)};

enum e_dtype {
    FLOAT32,
    FLOAT64,
    FLOAT128,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64
};

/**
 * @brief Matrix class:
 * About operator overloading:
 *  {+, -, *, /} Will create a new matrix and return it
 *  {+=, -=, *=, /=} Will modify the current matrix in place
 * Tips:
 *  A = A + B; // It will create a new matrix and copy the whole data from R-value to L-value
 *  A += B; // It will directly add B to A without dumb copying
 */
template<typename T>
class Matrix {

    /**
     * @brief Matrix class:
     * - A matrix class that can be used for any data type
     * - The data type is defined by the template parameter, T
     */
    Matrix(uint32_t rows, uint32_t cols) {
        const uint32_t _rows = rows;
        const uint32_t _cols = cols;
        try {
            T* _data = (T*)malloc(rows * cols * _dtype_size);
        } catch (std::bad_alloc& ba) {
            std::cerr << "bad_alloc caught: " << ba.what() << std::endl;
        }
    }

    ~Matrix() {
        free(_data);
    }

    void zeros() {
        fill((T)0);
    }

    void identity() {
        if (_rows != _cols) {
            throw std::invalid_argument("Identity matrix must be square");
        }
        fill((T)0);
        for (int i = 0; i < _rows; i++) {
            data[i * _cols + i] = (T)1;
        }
    }

    void fill(T value) {
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] = value;
        }
    }

    void Transpose() {
        Matrix<T> result(_cols, _rows);
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < _cols; j++) {
                result.data[j * _rows + i] = data[i * _cols + j];
            }
        }
        *this = result;
    }

    Matrix<T> operator+(Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, _cols);
        for (int i = 0; i < _rows * _cols; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Matrix<T> operator-(Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, _cols);
        for (int i = 0; i < _rows * _cols; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Matrix<T> operator*(Matrix<T> &other) {
        if (_cols != other._rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, other._cols);
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < other._cols; j++) {
                for (int k = 0; k < _cols; k++) {
                    result.data[i * other._cols + j] += data[i * _cols + k] * other.data[k * other._cols + j];
                }
            }
        }
        return result;
    }

    Matrix<T> operator*(T scalar) {
        Matrix<T> result(_rows, _cols);
        for (int i = 0; i < _rows * _cols; i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    Matrix<T> operator/(T scalar) {
        Matrix<T> result(_rows, _cols);
        for (int i = 0; i < _rows * _cols; i++) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    void operator+=(Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] += other.data[i];
        }
    }

    void operator-=(Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] -= other.data[i];
        }
    }

    void operator*=(Matrix<T> &other) {
        if (_cols != other._rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, other._cols);
        for (int i = 0; i < _rows; i++) {
            for (int j = 0; j < other._cols; j++) {
                for (int k = 0; k < _cols; k++) {
                    result.data[i * other._cols + j] += data[i * _cols + k] * other.data[k * other._cols + j];
                }
            }
        }
        *this = result;
    }

    void operator*=(T scalar) {
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] *= scalar;
        }
    }

    void operator/=(T scalar) {
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] /= scalar;
        }
    }

    void operator=(Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (int i = 0; i < _rows * _cols; i++) {
            data[i] = other.data[i];
        }
    }

    T* data() {
        return _data;
    }
};

/*** ACTIVATIONS ***/

template<typename T>
void ReLU(Matrix<T> &input, Matrix<T> &output) {
    // ReLU activation function
    // ...
    throw std::runtime_error("ReLU not implemented");
}

template<typename T>
void Sigmoid(Matrix<T> &input, Matrix<T> &output) {
    // Sigmoid activation function
    // ...
    throw std::runtime_error("Sigmoid not implemented");
}

template<typename T>
void Softmax(Matrix<T> &input, Matrix<T> &output) {
    // Softmax activation function
    // ...
    throw std::runtime_error("Softmax not implemented");
}

/*** LOSS ***/
template<typename T>
void CrossEntropy(Matrix<T> &input, Matrix<T> &output) {
    // Cross-entropy loss function
    // ...
    throw std::runtime_error("Cross-entropy not implemented");
}

template<typename T>
void MSE(Matrix<T> &input, Matrix<T> &output) {
    // Mean squared error loss function
    // ...
    throw std::runtime_error("MSE not implemented");
}


/**@brief
 * Multi-layer perceptron class workflow:
 *  - Add layers as you want, at any step
 */
template<typename T>
class MLP {
private:

    enum e_activation_method {
        RELU,
        SIGMOID,
        SOFTMAX
    }

    struct Layer<T> {
        uint32_t width;
        enum e_activation_method activation_method;
        Matrix<T> weights; // 2D matrix for weights of each link from previous layer (or input) to current layer
        Matrix<T> biases;  // 1D matrix for biases of each neuron in the current layer
    };

public:
    MLP(uint32_t input_size, uint32_t output_size) {
        // Initialize weights and biases
        // ...
        std::vector<struct Layer<T>> linear_layers;

    }

...    void add_layer_linear(uint32_t width, uint32_t index_at=-1) {
        // Add a layer to the network
        // ...
    }

    T* forward(T* input) {
        // Forward pass
        // ...
        return output;
    }

    void backward(T* input, T* output) {
        // Backward pass
        // ...
    }

    void update() {
        // Update weights and biases
        // ...
    }
};
