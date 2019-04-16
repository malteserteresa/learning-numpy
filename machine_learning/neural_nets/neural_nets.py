import numpy as np

size = 300
n_input_nodes = 2
n_hidden_nodes = 4
n_output_nodes = 1

learning_rate = 0.01


def generate_data(size, nodes_input):
    """Creates set of data points which are randomly labelled

    Args:
        int : The number of data points in to be created

    Returns:
        tuple of np arrays : An size x 2 numpy array of random floats and a size x 1 numpy array of labels

    """
    np.random.seed(422)  # so same numbers
    X = np.random.normal(0.0, 1.0, (size, nodes_input))
    Y = np.random.randint(2, size=size)
    return X, Y


def test_data_points_shape():
    assert generate_data(size, n_input_nodes)[0].shape == (300, 2)


def test_labels():
    assert generate_data(size, n_input_nodes)[1].shape == (300,)


def initialize_weights_and_biases(nodes_input, nodes_hidden, nodes_output):
    """Returns the initalized weights and biases from each layer in the net. They are randomly initialized and the
    biases set to zero"""
    W1 = np.random.randn(nodes_hidden, nodes_input) * 0.01
    b1 = np.zeros(shape=(nodes_hidden, 1))
    W2 = np.random.randn(nodes_output, nodes_hidden) * 0.01
    b2 = np.zeros(shape=(nodes_output, 1))
    return W1, b1, W2, b2


def test_first_layer_weights():
    assert initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)[0].shape == (4, 2)


def test_first_layer_biases():
    assert initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)[1].shape == (4, 1)


def test_second_layer_weights():
    assert initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)[2].shape == (1, 4)


def test_second_layer_biases():
    assert initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)[3].shape == (1, 1)


W1, b1, W2, b2 = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)

input_layer = generate_data(size, n_input_nodes)[0]
output_layer = generate_data(size, n_input_nodes)[1]

Z1 = np.dot(W1, input_layer.T) + b1 # first calc
A1 = np.tanh(Z1) # first activation layer
Z2 = np.dot(W2, A1) + b2 # second hypothesis
A2 = sigmoid(Z2, True)

# Compute loss

# Back propagation
dZ2 = A2 - output_layer
dW2 = (1/size) * np.dot(dZ2, A1.T)
db2 = (1 / size) * np.sum(dZ2, axis=1, keepdims=True)

dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
dW1 = (1 / size) * np.dot(dZ1, input_layer)
db1 = (1 / size) * np.sum(dZ1, axis=1, keepdims=True)

# update parameters
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
