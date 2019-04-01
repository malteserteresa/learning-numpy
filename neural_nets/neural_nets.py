import numpy as np
from sigmoid import sigmoid
from cross_entropy_cost import compute_cost


size = 300
learning_rate = 0.01

def create_data(size):
    """ Creates set of data points with a gaussian distribution and there labels
    """
    np.random.seed(422) # so same numbers
    X = np.random.normal(0.0, 1.0, (size,2))
    Y = np.random.randint(2, size=size)
    return X, Y

def test_data_points_shape():
    assert create_data(size)[0].shape == (300, 2)

def test_labels():
    assert create_data(size)[1].shape == (300,)



n_input = 2 # size of input layer`
n_hidden = 4
n_output = 1 # size of output layer

W1 = np.random.randn(n_hidden,n_input) * 0.01
b1 = np.zeros(shape=(n_hidden, 1))
W2 = np.random.randn(n_output,n_hidden) * 0.01
b2 = np.zeros(shape=(n_output, 1))


input_layer = create_data(size)[0]
output_layer = create_data(size)[1]

Z1 = np.dot(W1, input_layer.T) + b1 # first calc
A1 = np.tanh(Z1) # first activation layer
Z2 = np.dot(W2, A1) + b2 # second hypothesis
A2 = sigmoid(Z2, True)


dZ2 = A2 - output_layer
dW2 = (1/size) * np.dot(dZ2, A1.T)
db2 = (1 / size) * np.sum(dZ2, axis=1, keepdims=True)

dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
dW1 = (1 / size) * np.dot(dZ1, input_layer)
db1 = (1 / size) * np.sum(dZ1, axis=1, keepdims=True)

W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

# update parameters