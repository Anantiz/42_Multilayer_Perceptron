#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <sstream>
#include <immintrin.h>  // AVX and SSE; SIMD intrinsics

#define MATRIX_ALLIGNEMENT_SIZE 32

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
private:
    uint32_t _rows;
    uint32_t _cols;
    uint32_t _size;
    bool _cleanup;
    const size_t _dtype_size = sizeof(T);
    T* _data;

public:

    Matrix() : _rows(0), _cols(0), _size(0), _data(nullptr) {
        _cleanup = true;
    }

    /**
     * @brief Matrix class:
     * - A matrix class that can be used for any data type
     * - The data type is defined by the template parameter, T
     * :param: rows
     * :param: cols
     */
    Matrix(uint32_t rows, uint32_t cols) : _rows(rows), _cols(cols), _size(rows * cols) {
        _cleanup = true;
        _data = (T*)aligned_alloc(MATRIX_ALLIGNEMENT_SIZE, _size * _dtype_size);
        if (_data == nullptr) {
            throw std::bad_alloc();
        }
    }

    Matrix(const Matrix<T> &other) : _rows(other._rows), _cols(other._cols), _size(other._size) {
        _cleanup = true;
        _data = (T*)aligned_alloc(MATRIX_ALLIGNEMENT_SIZE, _size * _dtype_size);
        if (_data == nullptr) {
            throw std::bad_alloc();
        }
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] = other._data[i];
        }
    }

    Matrix(const std::string& string_repr) {
        std::istringstream iss(string_repr);
        iss >> _rows >> _cols;
        _size = _rows * _cols;
        _cleanup = true;
        _data = (T*)aligned_alloc(MATRIX_ALLIGNEMENT_SIZE, _size * _dtype_size);
        if (_data == nullptr) {
            throw std::bad_alloc();
        }
        for (unsigned int i = 0; i < _size; i++) {
            iss >> _data[i];
        }
    }

    /**
     * !! DANGEROUS !!
     * Constructor from raw data, takes ownership of the data
     * (If only a language existed where the compiler would enforce data ownership to avoid mistakes)
     * الحمد لله، أنا الآن مالك في روح Rust!
    */
    Matrix(uint32_t rows, uint32_t cols, T* data) : _rows(rows), _cols(cols), _size(rows * cols) {
        _cleanup = true;
        _data = data; // Take ownership of the data pointer
    }

    Matrix(Matrix<T> &&other) noexcept : _rows(other._rows), _cols(other._cols), _size(other._size) {
        _cleanup = other._cleanup;
        _data = other._data;
        other._data = nullptr;  // *Pirate noises*
    }

    Matrix<T> &operator=(const Matrix<T> &other) { // double && to allow r-value assignment
        if (this == &other) return *this;
        T* new_buff = _data;
        if (_cleanup) {
            // Don't reallocate if the size is the same
            if (_size != other._size) {
                new_buff = (T*)aligned_alloc(MATRIX_ALLIGNEMENT_SIZE, other._size * _dtype_size);
                if (new_buff == nullptr)
                    throw std::bad_alloc();
            }
        }
        else {
            // Auto-cleanup is disabled, we can't take reuse the buffer
            new_buff = (T*)aligned_alloc(MATRIX_ALLIGNEMENT_SIZE, _size * _dtype_size);
            if (new_buff == nullptr)
                throw std::bad_alloc();
        }
        for (unsigned int i = 0; i < other._rows * other._cols; i++) {
            new_buff[i] = other._data[i];
        }
        if (new_buff != _data)
            free(_data);
        _data = new_buff;
        _rows = other._rows;
        _cols = other._cols;
        _size = other._size;
        return *this;
    }

    /**@brief
     * Matrix class destructor
     * - Free the memory if the cleanup flag is set
     * Sometimes we want to isolate the data from the matrix object for performance reasons
     * In that case the matrix should not free the data
     */
    ~Matrix() {
        if (_cleanup)
            free(_data);
    }

    #ifdef __AVX__
    #undef __AVX__ // Cuz I can't make it not crash
    #endif

    void set_cleanup(bool free_data) {
        _cleanup = free_data;
    }

    Matrix<T> & zeros() {
        fill((T)0);
        return *this;
    }

    Matrix<T> &identity() {
        if (_rows != _cols) {
            throw std::invalid_argument("Identity matrix must be square");
        }
        fill((T)0);
        for (unsigned int i = 0; i < _rows; i++) {
            _data[i * _cols + i] = (T)1;
        }
        return *this;
    }

    Matrix<T> &fill(T value) {
        for (unsigned int i = 0; i < _size; i++) {
            _data[i] = value;
        }
        return *this;
    }

    /**
     * @brief
     * Fill the matrix with random values between 0 and 1
     */
    Matrix<T> &randomize() {
        for (unsigned int i = 0; i < _size; i++) {
            _data[i] = (T)rand() / RAND_MAX;
        }
        return *this;
    }

    Matrix<T> &randomize(T min, T max) {
        for (unsigned int i = 0; i < _size; i++) {
            _data[i] = min + (T)rand() / RAND_MAX * (max - min);
        }
        return *this;
    }

    Matrix<T> &xavier_init(double n) {
        // A nice randomizer for sigmoid activation
        T scale = sqrt(n / (_rows + _cols));
        return randomize(-scale, scale);
    }

    /**
     * @brief
     * Transpose the matrix in place
     */
    Matrix<T> &transpose_inplace() {
        Matrix<T> result(_cols, _rows);
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < _cols; j++) {
                result._data[j * _rows + i] = _data[i * _cols + j];
            }
        }
        *this = result;
        return *this;
    }

    /**
     * @brief
     * Transpose the matrix and return a new matrix
     */
    Matrix<T> as_transposed() const {
        Matrix<T> result(_cols, _rows);
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < _cols; j++) {
                result._data[j * _rows + i] = _data[i * _cols + j];
            }
        }
        return result;
    }

    /**
     * ACTUALLY  THIS DOESNT WORK !!!
    */
    void MultAddf(const Matrix<float> &mult, const Matrix<float> &add) {
        if (_rows != mult._cols || mult._rows != add._rows) {
            print_dim("this");
            mult.print_dim("mult");
            add.print_dim("add");
            throw std::invalid_argument("MultAddf: Matrix dimensions must match");
        }
        Matrix<float> result(_rows, mult._cols);
        #ifdef __AVX__
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < mult._cols; j++) {
                __m256 sum = _mm256_setzero_ps();  // Initialize AVX register to 0

                // Process 8 elements at a time
                unsigned int k = 0;
                unsigned int stop = _cols - (_cols % 8); // Don't underflow the uint
                for (; k < stop; k += 8) {
                    // Load 8 elements from input and weights
                    __m256 input_vec = _mm256_load_ps(&_data[i * _cols + k]);
                    __m256 weight_vec = _mm256_load_ps(&mult._data[k * mult._cols + j]);

                    // Multiply and accumulate
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(input_vec, weight_vec));
                }

                // Horizontal sum of the AVX register
                __m128 low = _mm256_extractf128_ps(sum, 0);
                __m128 high = _mm256_extractf128_ps(sum, 1);
                low = _mm_add_ps(low, high);
                low = _mm_hadd_ps(low, low);
                low = _mm_hadd_ps(low, low);

                // Add the remaining elements (if any)
                float total = _mm_cvtss_f32(low);
                for (; k < _cols; k++) {
                    total += _data[i * _cols + k] * mult._data[k * mult._cols + j];
                }

                // Add bias
                result._data[i * mult._cols + j] = total + add._data[j];
            }
        }
        # else
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < mult._cols; j++) {
                for (unsigned int k = 0; k < _cols; k++) {
                    result._data[i * mult._cols + j] += _data[i * _cols + k] * mult._data[k * mult._cols + j];
                }
                result._data[i * mult._cols + j] += add._data[i];
            }
        }
        free(_data);
        this->_data = result._data;
        this->_cols = result._cols;
        this->_rows = result._rows;
        this->_size = result._size;
        #endif
    }

    /**
     * @brief

     * Designed for updating the weights and biases in a neural network
     * :param mult: The matrix to multiply with, shall be mn*np compatible
     * :param add: The vector of biases, shall be mp compatible (p is 1)
     */
    void MultAdd(const Matrix<T> &mult, const Matrix<T> &add) {
        if (_cols != mult._rows || _rows != add._rows) {
            throw std::invalid_argument("MultAdd: Matrix dimensions must match");
        }
        Matrix<T> result(_rows, mult._cols);
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < mult._cols; j++) {
                for (unsigned int k = 0; k < _cols; k++) {
                    result._data[i * mult._cols + j] += _data[i * _cols + k] * mult._data[k * mult._cols + j];
                }
                result._data[i * mult._cols + j] += add._data[i];
            }
        }
        free(_data);
        this->_data = result._data;
        this->_cols = result._cols;
        this->_rows = result._rows;
        this->_size = result._size;
    }

    Matrix<T> operator+(const Matrix<T> &other) const {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] + other._data[i];
        }
        return result;
    }

    Matrix<T> operator-(const Matrix<T> &other) const {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] - other._data[i];
        }
        return result;
    }

    Matrix<T> operator*(const Matrix<T> &other) const {
        if (_cols != other._rows) {
            throw std::invalid_argument("Mult: Matrix dimensions must match");
        }

        Matrix<T> result(_rows, other._cols);

        #ifdef __AVX__
        if (_cols % 8 != 0) { // Somewhatof a fix but doesn't profit from AVX
            for (unsigned int i = 0; i < _rows; i++) {
                for (unsigned int j = 0; j < other._cols; j++) {
                    T sum = 0;
                    for (unsigned int k = 0; k < _cols; k++) {
                        sum += _data[i * _cols + k] * other._data[k * other._cols + j];
                    }
                    result._data[i * other._cols + j] = sum;
                }
            }
        } else {
            const int block_size = 8;  // AVX works well with blocks of 8 floats
            for (unsigned int i = 0; i < _rows; i++) {
                for (unsigned int j = 0; j < other._cols; j += block_size) {
                    __m256 sum_vec = _mm256_setzero_ps();  // Holds 8 summed values
                    for (unsigned int k = 0; k < _cols; k++) {
                        __m256 a_vec = _mm256_set1_ps(_data[i * _cols + k]);  // Broadcast A[i, k]
                        __m256 b_vec = _mm256_load_ps(&other._data[k * other._cols + j]);  // Load B[k, j:j+7]
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);  // sum += A[i,k] * B[k,j]
                    }
                    _mm256_store_ps(&result._data[i * other._cols + j], sum_vec);
                }
            }
        }
        #else
        // Standard triple-loop multiplication
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < other._cols; j++) {
                T sum = 0;
                for (unsigned int k = 0; k < _cols; k++) {
                    sum += _data[i * _cols + k] * other._data[k * other._cols + j];
                }
                result._data[i * other._cols + j] = sum;
            }
        }
        #endif
        return result;
    }

    Matrix<T> operator+(T scalar) const {
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] + scalar;
        }
        return result;
    }

    Matrix<T> operator-(T scalar) const {
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] - scalar;
        }
        return result;
    }

    Matrix<T> operator*(T scalar) const {
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] * scalar;
        }
        return result;
    }

    Matrix<T> operator/(T scalar) const {
        Matrix<T> result(_rows, _cols);
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            result._data[i] = _data[i] / scalar;
        }
        return result;
    }

    Matrix<T> &operator+=(const Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] += other._data[i];
        }
        return *this;
    }

    Matrix<T> &operator-=(const Matrix<T> &other) {
        if (_rows != other._rows || _cols != other._cols) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] -= other._data[i];
        }
        return *this;
    }

    Matrix<T> &operator*=(const Matrix<T> &other) {
        if (_cols != other._rows) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        Matrix<T> result(_rows, other._cols);
        for (unsigned int i = 0; i < _rows; i++) {
            for (unsigned int j = 0; j < other._cols; j++) {
                for (unsigned int k = 0; k < _cols; k++) {
                    result._data[i * other._cols + j] += _data[i * _cols + k] * other._data[k * other._cols + j];
                }
            }
        }
        *this = result;
        return *this;
    }

    Matrix<T> &operator+=(T scalar) {
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] += scalar;
        }
        return *this;
    }

    Matrix<T> &operator-=(T scalar) {
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] -= scalar;
        }
        return *this;
    }

    Matrix<T> &operator*=(T scalar) {
        #ifdef __AVX__
        if (_size % 8 == 0) {
            unsigned int i = 0;
            for (; i <= _size - 8; i += 8) {
                __m256 input_vec = _mm256_load_ps(&_data[i]);
                __m256 scalar_vec = _mm256_set1_ps(scalar);
                __m256 sum = _mm256_mul_ps(input_vec, scalar_vec);
                _mm256_store_ps(&_data[i], sum);
            }
            while (i < _size)
            _data[i++] *= scalar;
        } else {
            for (unsigned int i = 0; i < _size; i++) {
                _data[i] *= scalar;
            }
        }
        #else
        for (unsigned int i = 0; i < _size; i++) {
            _data[i] *= scalar;
        }
        #endif
        return *this;
    }

    Matrix<T> &operator/=(T scalar) {
        #ifdef __AVX__
        if (_size % 8 == 0) {
            unsigned int i = 0;
            for (; i <= _size - 8; i += 8) {
                __m256 input_vec = _mm256_load_ps(&_data[i]);
                __m256 scalar_vec = _mm256_set1_ps(scalar);
                __m256 sum = _mm256_div_ps(input_vec, scalar_vec);
                _mm256_store_ps(&_data[i], sum);
            }
            while (i < _size)
            _data[i++] /= scalar;
        } else {
            for (unsigned int i = 0; i < _size; i++) {
                _data[i] /= scalar;
            }
        }
        #else
        for (unsigned int i = 0; i < _size; i++) {
            _data[i] /= scalar;
        }
        #endif
        return *this;
    }

    T &operator[](uint32_t index) {
        if (index >= _size) {
            throw std::out_of_range("Matrix index out of range");
        }
        return _data[index];
    }

    T operator[](uint32_t index) const {
        if (index >= _size) {
            throw std::out_of_range("Matrix index out of range");
        }
        return _data[index];
    }

    inline T* data() {
        return _data;
    }

    inline uint32_t rows() const {
        return _rows;
    }

    inline uint32_t cols() const {
        return _cols;
    }

    inline uint32_t size() const {
        return _size;
    }

    void print_dim(const std::string &text="") const {
        std::cout << text << ": [r " << _rows << ", c " << _cols << "]" << std::endl;
    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
        for (unsigned int i = 0; i < matrix._rows; i++) {
            for (unsigned int j = 0; j < matrix._cols; j++) {
                os << matrix._data[i * matrix._cols + j] << " ";
            }
            os << std::endl;
        }
        return os;
    }

    Matrix<T> &sigmoid_inplace() {
        for (unsigned int i = 0; i < _rows * _cols; i++) {
            _data[i] = 1 / (1 + exp(-_data[i]));
        }
        return *this;
    }

    /**
     * @brief
     * WARNING: Outputs bullshit if the input is not a vector
     */
    Matrix<T> as_diagonal() const {
        Matrix<T> result(_size, _size);
        result.fill((T)0);
        for (unsigned int i = 0; i < _size; i++) {
            result._data[i * _size + i] = _data[i];
        }
        return result;
    }

    static Matrix<T> hadamard(const Matrix<T> &a, const Matrix<T> &b) {
        if (a._rows != b._rows || a._cols != b._cols) {
            throw std::invalid_argument("hadamard:Matrix dimensions must match");
        }
        Matrix<T> result(a._rows, a._cols);
        for (unsigned int i = 0; i < a._rows * a._cols; i++) {
            result._data[i] = a._data[i] * b._data[i];
        }
        return result;
    }

    /**
     * @brief
     * Scales each row of the matrix with the corresponding element of the vector
     * The vector must have the same number of rows as the matrix
     */
    static Matrix<T> matrix_row_scaling_from_vector(const Matrix<T> &matrix, const Matrix<T> &vector) {
        if (matrix._rows != vector._rows) {
            throw std::invalid_argument("row scaling:Rows dimensions must match");
        }
        Matrix<T> result(matrix._rows, matrix._cols);
        for (unsigned int i = 0; i < matrix._size; i++) {
            result._data[i] = matrix._data[i] * vector._data[i % vector._rows];
        }
        return result;
    }

    void set_at(uint32_t row, uint32_t col, T value) {
        if (row >= _rows || col >= _cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        _data[row * _cols + col] = value;
    }

    void set_at_flat(uint32_t index, T value) {
        if (index >= _size) {
            throw std::out_of_range("Matrix index out of range");
        }
        _data[index] = value;
    }

    T get_at(uint32_t row, uint32_t col) const {
        if (row >= _rows || col >= _cols) {
            throw std::out_of_range("Matrix index out of range");
        }
        return _data[row * _cols + col];
    }

    T get_at_flat(uint32_t index) const {
        if (index >= _size) {
            throw std::out_of_range("Matrix index out of range");
        }
        return _data[index];
    }

    /**
     * @brief
     * Returns a string representation of the matrix to save it to a file
     * Can be re-imported with the string constructor
     */
    std::string export_string() {
        std::string result = std::to_string(_rows) + " " + std::to_string(_cols) + " ";
        for (unsigned int i = 0; i < _size; i++) {
            result += std::to_string(_data[i]) + " ";
        }
        return result;
    }

    T argmax_index() const {
        T max = _data[0];
        T index = 0;
        for (unsigned int i = 1; i < _size; i++) {
            if (_data[i] > max) {
                max = _data[i];
                index = i;
            }
        }
        return index;
    }

    T argmax_value() const {
        T max = _data[0];
        for (unsigned int i = 1; i < _size; i++) {
            if (_data[i] > max) {
                max = _data[i];
            }
        }
        return max;
    }

    T dot(const Matrix<T> &other) const {
        if (_size != other._size) {
            throw std::invalid_argument("dot:Matrix dimensions must match");
        }
        T result = 0;
        for (unsigned int i = 0; i < _size; i++) {
            result += _data[i] * other._data[i];
        }
        return result;
    }
};
