#!python3
import mlp_module
import matplotlib.pyplot as plt
import numpy as np

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

def plot_report(report):
    loss = report.train_loss

    # Create a single figure with two subplots
    plt.figure(figsize=(12, 6))

    # Plot accuracy over epochs
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(np.arange(1, len(report.train_accuracy) + 1), report.train_accuracy, label="Train")
    plt.plot(np.arange(1, len(report.test_accuracy) + 1), report.test_accuracy, label="Test")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss over epochs
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(np.arange(1, len(report.train_loss) + 1), report.train_loss, label="Train")
    plt.plot(np.arange(1, len(report.test_loss) + 1), report.test_loss, label="Test")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the figure
    plt.show()


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
        print(f"Saved weights to {save_weights_file_path}")

    except Exception as e:
        print(f"Failed to train MLP: {e}", file=sys.stderr)
        sys.exit(1)

def train_with_tests(epochs, mlp, training_set_file_path, test_set_file_path, save_weights_file_path):
    """
    For each epoch will run tests to see the accuracy of the model
    """
    if save_weights_file_path is None:
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_weights_file_path = f"./weights_{date}.mlp"

    if os.access(training_set_file_path, os.R_OK) is False:
        print(f"Training set file {training_set_file_path} is not readable", file=sys.stderr)
        sys.exit(1)
    if os.access(os.path.dirname(save_weights_file_path), os.W_OK) is False:
        print(f"Directory '{os.path.dirname(save_weights_file_path)}' is not writable", file=sys.stderr)
    try:
        report = mlp.train_test_earlystop(training_set_file_path, test_set_file_path, epochs)
        mlp.save_model(save_weights_file_path)
        print(f"Saved weights to {save_weights_file_path}")
        plot_report(report)
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
    parser.add_argument('--load_model_file', type=str, default=None, help='Path to load a weights file') # Group A
    parser.add_argument('--input_size', type=int, default=None, help='Number of input neurons, must be one per feature')
    parser.add_argument('--output_size', type=int, default=None, help='Number of output neurons, must be one per class')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--training_set_file', type=str, default=None, help='Path to the training set file')
    parser.add_argument('--save_model_file', type=str, default=None, help='Path to save the weights file')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for training')
    parser.add_argument('--test_set_file', type=str, default=None, help='Path to the test set file')
    parser.add_argument(
    "--hidden_layers",
    type=parse_hidden_layer,
    nargs="+",
    default=None,
    help="Hidden layers as space-separated tuples: (width, activation) (width, activation) ..."
    )
    args = parser.parse_args()


    if args.load_model_file is not None:
        print(f"Loading MLP from {args.load_model_file}")
        mlp = init_mlp_file(args.load_model_file)
        if args.test_set_file is not None:
            print("Evaluating MLP")
            mlp.test_from_file(args.test_set_file)
        else:
            print("No test set provided, exiting")
            sys.exit(1)
    else:
        if (args.input_size is None or args.output_size is None or args.epochs is None or args.hidden_layers is None):
            print("Missing required arguments for training, exiting")
            sys.exit(1)

        print(f"Training MLP with input size {args.input_size}, output size {args.output_size}, epochs {args.epochs},"
            f"training set file {args.training_set_file}, save weights file {args.save_model_file} and hidden layers:\n {args.hidden_layers}")
        mlp = init_mlp(args.input_size, args.output_size, args.learning_rate, args.hidden_layers)
        if (args.training_set_file is not None and args.test_set_file is not None):
            train_with_tests(args.epochs, mlp, args.training_set_file, args.test_set_file, args.save_model_file)
        elif args.training_set_file is not None:
            train(args.epochs, mlp, args.training_set_file, args.save_model_file)
        else:
            print("No training set provided and no model loaded to be tested, exiting")
            sys.exit(1)

if __name__ == '__main__':
    main()
