import numpy as np

def mode(arr):
    """Returns the most common value in an array."""
    most_frequent = {key: 0 for key in arr}
    for element in arr:
        most_frequent[int(element)] += 1

    return max(most_frequent, key=most_frequent.get)


def test_mode_numbers():
    numbers = np.array([1, 1, 1, 1, 1, 3])
    assert mode(numbers) == 1


def test_mode_string():
    strings = np.array(["cat", "rat", "cat"])
    assert mode(strings) == "cat"


def right_skew(min, max, n):
    """Returns an array of numbers between the given limits with numbers skewed to the right."""
    pers = np.arange(min, max, 1)
    # Make each of the last 41 elements 5x more likely
    prob = [1.0] * (len(pers) - 41) + [5.0] * 41
    # Normalising to 1.0
    prob /= np.sum(prob)
    return np.array([ np.random.choice(pers, 1, p=prob)[0] for n in range(0,n) ])