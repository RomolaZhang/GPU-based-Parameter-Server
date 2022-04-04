from collections import defaultdict
from email.policy import default
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_dense_data(file):
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    return dataset

def parse_line(line):
    """Parse a line of training data file to produce a data sample record."""
    parts = line.split()
    label = int(parts[0])
    # the program requires binary labels in {0, 1}
    # the dataset may have binary labels -1 and 1, we convert all -1 to 0
    label = 0 if label == -1 else label
    feature_ids = []
    feature_vals = []
    for part in parts[1:]:
        feature = part.split(":")
        # the datasets have feature ids in [1, N] we convert them
        # to [0, N - 1] for array indexing
        feature_ids.append(int(feature[0]) - 1)
        feature_vals.append(float(feature[1]))
    return (label, (np.array(feature_ids), np.array(feature_vals, dtype=np.float64)))


def load_sparse_data(file):
    dataset = []
    labels = []
    for line in open(file, "r").readlines():
        label, data = parse_line(line)
        dataset.append(data)
        labels.append(label)
    return dataset, labels

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def l(theta, X, y):
    J = -y * (X @ theta) + np.log(1 + np.exp(X @ theta))
    return np.mean(J)

def train_dense(X, y, num_epoch, learning_rate):
    theta = np.zeros(X.shape[1]) # b + M features
    N = X.shape[0]
    for _ in range(num_epoch):
        for i in range(N):
            g = X[i] * (y[i] - sigmoid(X[i] @ theta))
            theta = theta + learning_rate * g
    return theta


def forward_sparse(theta, X, i):
    feature_ids, feature_vals = X[i]
    weights = []
    for f_id in feature_ids:
        weights.append(theta.get(f_id, 0))
    weights = np.array(weights, dtype=np.float64)
    pred = sigmoid(feature_vals.dot(weights))
    return pred


def train_sparse(X, y, num_epoch, learning_rate):
    theta = defaultdict(float)
    N = len(X)
    for _ in range(num_epoch):
        for i in range(N):
            feature_ids, feature_vals = X[i]
            y_hat = forward_sparse(theta, X, i)
            diff = y[i] - y_hat
            gradient = diff * feature_vals

            for i in range(len(feature_ids)):
                theta[feature_ids[i]] += learning_rate * gradient[i]
    return theta


def predict(theta, X):
    if not isinstance(theta, dict):
        predicted = sigmoid(X @ theta) >= 0.5
        return predicted.astype(np.int64)

    N = len(X)
    predicted = []
    for i in range(N):
        predicted.append(forward_sparse(theta, X, i) >= 0.5)
    
    return np.array(predicted, dtype=np.int64)


def compute_error(y_pred, y):
    accuracy = (y_pred == y).mean()
    return 1 - accuracy



if __name__ == '__main__':
    metric_type = sys.argv[2]
    num_epoch = int(sys.argv[3])
    learning_rate = float(sys.argv[4])

    if metric_type == "dense":
        train_X = load_dense_data(sys.argv[1]) # shape(N,)
        train_y = np.copy(train_X[:, 0])
        train_X[:, 0] = 1

        theta = train_dense(train_X, train_y, num_epoch, learning_rate)
    else:
        train_X, train_y = load_sparse_data(sys.argv[1]) # shape(N,)
        theta = train_sparse(train_X, train_y, num_epoch, learning_rate)


    train_predicted = predict(theta, train_X)
    print("error(train): {:.6f}\n".format(compute_error(train_predicted , train_y)))

# python3 src/python_scripts/lr.py data/formatted_small_train_sparse.tsv sparse 500 0.00001
# python3 src/python_scripts/lr.py data/formatted_small_train_dense.tsv dense 500 0.00001

# python3 src/python_scripts/lr.py data/formatted_large_train_dense.tsv dense 500 0.00001
# error(train): 0.042500

# python3 src/python_scripts/lr.py data/formatted_small_train_sparse.tsv sparse 500 0.00001
# error(train): 0.042500