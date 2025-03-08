#include "headers/mlp.hpp"
#include "headers/matrix.hpp"

int main() {
    try {
        float lr = 0.01;
        int input_size = 3;
        int output_size = 2;
        std::vector<std::pair<uint32_t, uint32_t>> hidden_layers;

        hidden_layers.push_back(std::pair<uint32_t, uint32_t>(input_size, (uint32_t)Mlp<float>::RELU)); // l1
        hidden_layers.push_back(std::pair<uint32_t, uint32_t>(8, (uint32_t)Mlp<float>::RELU)); // l2
        hidden_layers.push_back(std::pair<uint32_t, uint32_t>(16, (uint32_t)Mlp<float>::RELU)); // l2
        hidden_layers.push_back(std::pair<uint32_t, uint32_t>(output_size, (uint32_t)Mlp<float>::SOFTMAX)); // l3


        Mlp<float> mlp = Mlp<float>(input_size, output_size, lr, hidden_layers);

        int true_label_index;
        Matrix<float> mock_input = Matrix<float>(3, 1);

        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 0;
        Matrix<float> nn_result = mlp.forward(mock_input, false);
        std::cout << "No training" << std::endl;
        std::cout << " Predict A: " << nn_result[0] << " " << nn_result[1] << std::endl;
        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 1;
        nn_result = mlp.forward(mock_input, false);
        std::cout << " Predict B: " << nn_result[0] << " " << nn_result[1] << std::endl;

        // Make bullshit data
        std::vector<std::pair<Matrix<float>, size_t>> data;
        data.reserve(300);
        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 0;
        for (int i = 0; i < 150; i++) {
            data.push_back(std::pair<Matrix<float>, size_t>(mock_input, 0));
        }
        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 1;
        for (int i = 0; i < 150; i++) {
            data.push_back(std::pair<Matrix<float>, size_t>(mock_input, 1));
        }

        mlp.train(500, data);


        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 0;
        nn_result = mlp.forward(mock_input, false);
        std::cout << "\nAfter training"<< std::endl;
        std::cout << " Predict A: " << nn_result[0] << " " << nn_result[1] << std::endl;
        mock_input[0] = 1;
        mock_input[1] = 0;
        mock_input[2] = 1;
        nn_result = mlp.forward(mock_input, false);
        std::cout << " Predict B: " << nn_result[0] << " " << nn_result[1] << std::endl;

    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return (0);
}