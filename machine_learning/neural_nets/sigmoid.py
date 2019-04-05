import numpy as np

def sigmoid(x, derivative=False):
    """The activation function for the output layer of the neural net."""
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def test_sigmoid():
    assert sigmoid(0, False) == 0.5

def test_max():
    assert round(sigmoid(6, False)) == 1

def test_min():
    assert round(sigmoid(-6, False)) == 0
