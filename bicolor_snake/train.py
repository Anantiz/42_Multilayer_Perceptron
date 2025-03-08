import mlp_module
import argparse
from datetime import datetime
import sys
import os

RELU = 'relu'
SIGMOID = 'sigmoid'
SOFTMAX = 'softmax'
mapping = { # Cuz exporting enums is hard, ♫I'm only human after all ♫, don't put the blame on me♫♫
    RELU: 0,
    SIGMOID: 1,
    SOFTMAX: 2
}

def init_mlp(input_size, output_size, learning_rate, hidden_layers:list[tuple[int, int]]):
    if input_size <= 0 or output_size <= 0:
        raise ValueError("Input and output size must be positive integers")
    if not hidden_layers:
        raise ValueError("At least one hidden layer must be defined")
    try:
        return mlp_module.Mlp(input_size, output_size, learning_rate, hidden_layers)
    except Exception as e:
        print(f"Failed to create Mlp: {e}", file=sys.stderr)
        sys.exit(1)

def init_mlp_file(path):
    try:
        mlp = mlp_module.Mlp(path)
        if mlp is None:
            raise Exception("Failed to load Mlp")
        return mlp
    except Exception as e:
        print(f"Failed to load MLP: {e}", file=sys.stderr)
        sys.exit(1)

def train(epochs, mlp, training_set_file_path, save_weights_file_path):
    if save_weights_file_path is None:
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_weights_file_path = f"./weights_{date}.mlp"

    if os.access(training_set_file_path, os.R_OK) is False:
        print(f"Training set file {training_set_file_path} is not readable", file=sys.stderr)
        sys.exit(1)
    if os.access(os.path.dirname(save_weights_file_path), os.W_OK) is False:
        print(f"Directory '{os.path.dirname(save_weights_file_path)}' is not writable", file=sys.stderr)
    try:
        mlp.train_from_file(epochs, training_set_file_path)
        mlp.save_model(save_weights_file_path)
    except Exception as e:
        print(f"Failed to train MLP: {e}", file=sys.stderr)
        sys.exit(1)

### ARGUMENT PARSING ###

def parse_hidden_layer(s):
    try:
        width, act_type = s.strip("()").split(",")
        return (int(width), mapping.get(act_type.strip(), None))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid format for hidden layer: {s}. Expected format: (width, type)")

def main():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron')
    parser.add_argument('--input_size', type=int, required=True, help='Number of input neurons, must be one per feature')
    parser.add_argument('--output_size', type=int, required=True, help='Number of output neurons, must be one per class')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--training_set_file', type=str, required=True, help='Path to the training set file')
    parser.add_argument('--save_model_file', type=str, default=None, help='Path to save the weights file')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for training')
    parser.add_argument('--test_set_file', type=str, default=None, help='Path to the test set file')
    parser.add_argument(
    "--hidden_layers",
    type=parse_hidden_layer,
    nargs="+",
    required=True,
    help="Hidden layers as space-separated tuples: (width, activation) (width, activation) ..."
    )
    args = parser.parse_args()

    print(f"Training MLP with input size {args.input_size}, output size {args.output_size}, epochs {args.epochs},"
        f"training set file {args.training_set_file}, save weights file {args.save_model_file} and hidden layers:\n {args.hidden_layers}")

    mlp = init_mlp(args.input_size, args.output_size, args.learning_rate, args.hidden_layers)

    train(args.epochs, mlp, args.training_set_file, args.save_model_file)
    if args.test_set_file is not None:
        print("Evaluating MLP")
        mlp.test_from_file(args.test_set_file)

if __name__ == '__main__':
    main()
