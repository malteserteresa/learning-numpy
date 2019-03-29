import numpy as np
import visualization as v

def gaussian(x, mu, sigma):
    """ Implementation of Gaussian function
    """
    norm = 1/np.sqrt(2*np.pi)*sigma


    return norm * np.exp(-0.5*((x-mu)/sigma)**2)

x = np.linspace(-10, 10, 100)
v.plot(x, gaussian(x, 0, 2))