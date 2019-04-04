import numpy as np

def probability_density(p, r, n):
    """Probability density of binomial distribution.


    Args:
        p (float): likelihood of success
        r (int) : number of trials
        n (int) : number of successes

    Returns:
        float : likelihood of event occuring

    """
    return round(nCr(n, r)*p**r*(1-p)**(n-r), 4)


def nCr(n, r):
    """Gives the n objects, how many combinations of size r can be made
    """
    return int(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))


def test_combinations():
    """
    """
    assert nCr(52, 5) == 2598960

print(probability_density(0.5,10,1000))

def test_prob_density():
    assert probability_density(0.2, 2, 8) == 0.2936