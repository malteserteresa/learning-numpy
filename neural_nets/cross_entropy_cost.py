import numpy as np
import sys
sys.path.append('../')
import visualization


def cross_entropy(activated_output_layer, output_layer):
    """ Computes the cross-entropy cost for each layer
    """
    return np.multiply(np.log(activated_output_layer), output_layer) + np.multiply((1 - output_layer),
                                                                            np.log(1 - activated_output_layer))
def compute_cost(activated_output_layer, output_layer, size):
    """ Computes the total cost
    """
    return - np.sum(cross_entropy(activated_output_layer, output_layer, size))/size

