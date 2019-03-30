import numpy as np
import matplotlib.pyplot as plt

size = 300

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

n_x = 2 # size of input layer`
n_h = 4
n_y = 1 # size of output layer

W1 = np.random.randn(n_h,n_x) * 0.01
b1 = np.zeros(shape=(n_h, 1))
W2 = np.random.randn(n_y,n_h) * 0.01
b2 = np.zeros(shape=(n_y, 1))

#Visualize
input_layer = create_data(size)[0]
output_layer = create_data(size)[1]
plt.scatter(input_layer[:, 0], input_layer[:, 1], c=output_layer, s=40, cmap=plt.cm.Spectral);
plt.show()

# create data
# define network structure
# get weights and biases
# Iterate: forward prop, loss, backward prop, update parameters