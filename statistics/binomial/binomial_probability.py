from binomial_distribution import nCx

def b(n, p, x):
    """ Calculates the binomial distribution.

     Args:
        n (int) : The number of events
        p (float) : The likelihood of successful event
        x (int) : The number of outcomes of the event
    Returns:
        int : the probability of the successful event occurring
    """

    return nCx(n, x) * p ** x * (1 - p) ** (n - x)


