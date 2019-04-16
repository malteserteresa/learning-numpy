from binomial.binomial_expansion import nCr

def probability_density(p, r, n):
    """Probability density of binomial distribution.


    Args:
        p (float) : likelihood of success
        r (int) : number of trials
        n (int) : number of successes

    Returns:
        float : likelihood of event occurring

    """
    return round(nCr(n, r)*p**r*(1-p)**(n-r), 4)

print(probability_density(0.5,10,1000))

def test_prob_density():
    assert probability_density(0.2, 2, 8) == 0.2936