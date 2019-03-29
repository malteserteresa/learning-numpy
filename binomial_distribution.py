import numpy as np

def probability_density(p, r, n):
    """ Probability density of binomial distribution where p = likelihood of success, n number of successes r number of trials
    """
    return round(nCr(n, r)*p**r*(1-p)**(n-r), 4)


def nCr(n, r):
    """ Gives the n objects, how many combinations of size r can be made
    """
    return int(np.math.factorial(n)/(np.math.factorial(r)*np.math.factorial(n-r)))


def test_combinations():
    """ Given a deck of 52 playing cards, how many different hands can I have with 5 cards
    """
    assert nCr(52, 5) == 2598960

print(probability_density(0.5,10,1000))

def test_prob_density():
    assert probability_density(0.2, 2, 8) == 0.2936