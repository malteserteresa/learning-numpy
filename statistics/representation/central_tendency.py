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


