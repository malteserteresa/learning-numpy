import numpy as np

def cross_entropy(activated_output_layer, output_layer):
    """Computes the cross-entropy cost for each layer."""
    return np.multiply(np.log(activated_output_layer), output_layer) + np.multiply((1 - output_layer),
                                                                            np.log(1 - activated_output_layer))
def compute_cost(activated_output_layer, output_layer, size):
    """Computes the total cost."""
    return - np.sum(cross_entropy(activated_output_layer, output_layer))/size

def test_cross_entropy():
    X = np.random.normal(0.0, 1.0, (2, 300))
    Y =  np.random.randint(2, size=300)
    assert np.any(cross_entropy(X, Y) <0)