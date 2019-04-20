import numpy as np

from cross_entropy_cost import compute_cost
from sigmoid import sigmoid

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
    assert generate_data(size, n_input_nodes)[0].shape == (size, n_input_nodes)


def test_labels():
    assert generate_data(size, n_input_nodes)[1].shape == (size,)


def initialize_weights_and_biases(nodes_input, nodes_hidden, nodes_output):
    """Returns the initalized weights and biases from each layer in the net. They are randomly initialized and the
    biases set to zero.

    Args :
        n_input_nodes (int) : Number of features of the examples.
        n_hidden_nodes (int) : Number of units in the hidden layer.
        n_output_nodes (int) : Number of features of the targets.

    Returns:
       parameters (dict) : A dictionary containing the initialized parameters of the system
    """
    W1 = np.random.randn(nodes_hidden, nodes_input) * 0.01
    b1 = np.zeros(shape=(nodes_hidden, 1))
    W2 = np.random.randn(nodes_output, nodes_hidden) * 0.01
    b2 = np.zeros(shape=(nodes_output, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def test_first_layer_weights():
    W1 = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)["W1"]
    assert W1.shape == (n_hidden_nodes, n_input_nodes)
    assert np.all(W1 != 0)


def test_first_layer_biases():
    b1 = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)["b1"]
    assert b1.shape == (n_hidden_nodes, 1)
    assert np.all(b1 == 0)


def test_second_layer_weights():
    W2 = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)["W2"]
    assert W2.shape == (1, n_hidden_nodes)
    assert np.all(W2 != 0)


def test_second_layer_biases():
    b2 = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)["b2"]
    assert b2.shape == (1, 1)
    assert np.all(b2 == 0)


def feed_forward(parameters, input_layer):
    """
    Args:
        parameters (dict) : The parameters of the neural net (weights and biases)
        input_layer (numpy array) : The examples to learn the mapping of.
    Returns:
        parameters : A dictionary of the
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b1']

    Z1 = np.dot(W1, input_layer.T) + b1  # first calc
    A1 = np.tanh(Z1)  # first activation layer
    Z2 = np.dot(W2, A1) + b2  # second hypothesis
    A2 = sigmoid(Z2, True)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def setUp(size, n_input_nodes):
    """Returns the weights and biases to begin their journey in the neural network.
    """
    X, Y = generate_data(size, n_input_nodes)
    parameters = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)
    return feed_forward(parameters, X)


def test_feed_forward():
    calculations = setUp(size, n_input_nodes)

    assert calculations['Z1'].shape == (n_hidden_nodes, size)
    assert calculations['A1'].shape == (n_hidden_nodes, size)
    assert calculations['Z2'].shape == (n_hidden_nodes, size)
    assert calculations['A2'].shape == (n_hidden_nodes, size)


def test_feed_forward_not_null():
    calculations = setUp(size, n_input_nodes)

    assert np.all(calculations['Z1'] != 0)
    assert np.all(calculations['A1'] != 0)
    assert np.all(calculations['Z2'] != 0)
    assert np.all(calculations['A2'] != 0)


def backward_propagation(calculations, parameters, input_layer, output_layer, size):
    """Returns the results of gradient steepest decent.
    """

    A1 = calculations['A1']
    A2 = calculations['A2']

    W2 = parameters['W2']

    dZ2 = A2 - output_layer
    dW2 = (1 / size) * np.dot(dZ2, A1.T)
    db2 = (1 / size) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / size) * np.dot(dZ1, input_layer)
    db1 = (1 / size) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(parameters, gradients):
    """Returns the updated parameters after multiplying them by the gradients and learning rate.
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b1']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db1']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


parameters = initialize_weights_and_biases(n_input_nodes, n_hidden_nodes, n_output_nodes)

input_layer = generate_data(size, n_input_nodes)[0]
output_layer = generate_data(size, n_input_nodes)[1]

n_iterations = 1

for i in range(0, n_iterations):

    calculations = feed_forward(parameters, input_layer)
    print(calculations['A2'])

    cost = compute_cost(calculations['A2'], output_layer, size)

    gradients = backward_propagation(calculations, parameters, input_layer, output_layer, size)

    update_parameters(parameters, gradients)

    if i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))


