### DATA DATA DATA ###

import argparse
import pandas as pd

# The data is structured as follows:
# col 1 : id
# cold 2: label: M|B == malignant | benign

# We want to, create a training and test set, and preprocess the data
# 1. If a row has a missing value, we will remove it
# 2. We will convert the labels to and index: Index of the neuron that shall be activated for the given label

def split_data(data_path='data.csv', train_path='train_set.csv', test_path='test_set.csv',frac=0.75, mapping={'M': 0, 'B': 1}):
    data = pd.read_csv(data_path, header=None)
    data = data.dropna()
    data.iloc[:,1] = data.iloc[:,1].map(mapping).fillna(-1).astype(int)
    data.drop(data[data.iloc[:,1] == -1].index, inplace=True) # Remove rows with invalid labels
    train_data = data.sample(frac=frac)
    test_data = data.drop(train_data.index)
    train_data.to_csv(train_path, index=False, header=False)
    test_data.to_csv(test_path, index=False, header=False)

### ARGUMENT PARSING ###

def main():
    parser = argparse.ArgumentParser(description='Preprocess and split the data')
    parser.add_argument('--data_path', type=str, default='data.csv', help='Path to the data file')
    parser.add_argument('--train_path', type=str, default='train_set.csv', help='Path to the training data file')
    parser.add_argument('--test_path', type=str, default='test_set.csv', help='Path to the test data file')
    parser.add_argument('--frac', type=float, default=0.75, help='Fraction of the data to be used for training')
    args = parser.parse_args()

    split_data(args.data_path, args.train_path, args.test_path, args.frac)

if __name__ == '__main__':
    main()