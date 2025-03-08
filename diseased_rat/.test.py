from matrix_module import MatrixF32 as Matrix  # Import the MatrixF32 class from your compiled module
from mlp_module import Mlp  # Import the Mlp class from your compiled module

# Activation functions; too lazy to bind the cpp enum to python
RELU = 0
SIGMOID = 1
SOFTMAX = 2

def main():
    try:
        # Parameters
        lr = 0.01
        input_size = 3
        output_size = 2

        # Define hidden layers
        hidden_layers = [
            (input_size, RELU),  # Layer 1
            (8, RELU),           # Layer 2
            (16, RELU),          # Layer 3
            (output_size, SOFTMAX)  # Output layer
        ]

        # Create the MLP
        mlp = Mlp(input_size, output_size, lr, hidden_layers)

        # Create mock input
        mock_input_A = Matrix(input_size,1)
        mock_input_B = Matrix(input_size,1)

        # Test without training
        mock_input_A.set_at_flat(0, 1)
        mock_input_A.set_at_flat(1, 0)
        mock_input_A.set_at_flat(2, 0)
        nn_result = mlp.forward(mock_input_A, False)
        print("No training")
        print(f" Predict A: {nn_result[0]} {nn_result[1]}")

        mock_input_B.set_at_flat(0, 1)
        mock_input_B.set_at_flat(1, 0)
        mock_input_B.set_at_flat(2, 1)
        nn_result = mlp.forward(mock_input_B, False)
        print(f" Predict B: {nn_result[0]} {nn_result[1]}")


        # Train the MLP
        epochs = 1500
        for i in range(epochs):
            # Train with input A
            true_label_index = 0
            nn_result = mlp.forward(mock_input_A, True)
            mlp.backward(nn_result, mock_input_A, true_label_index)

            # Train with input B
            true_label_index = 1
            nn_result = mlp.forward(mock_input_B, True)
            mlp.backward(nn_result, mock_input_B, true_label_index)

        # Test after training
        nn_result = mlp.forward(mock_input_A, False)
        print("\nAfter training")
        print(f" Predict A: {nn_result[0]} {nn_result[1]}")

        nn_result = mlp.forward(mock_input_B, False)
        print(f" Predict B: {nn_result[0]} {nn_result[1]}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()