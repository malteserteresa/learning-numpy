# Pascals triangle

def pascals_triangle(order):
    """Prints pascals triangle.

    Args:
        int : The number of rows of pascals triangle to print

    """


def recursive_summation(split_row, first = None):
    """Returns the one half of the coefficients of pascals triangle for the next row.
    """
    if(first == None):
        first = []

    # BASE CASE list = [1]
    if(split_row == [1]):
        return split_row + first[::-1]
    else:
        first.append(split_row[-2]+split_row[-1])
        split_row = split_row[:-1]
        return recursive_summation(split_row, first)


def test_creates_half_of_next_row():
    assert recursive_summation([1, 3], None) == [1, 4]
    assert recursive_summation([1, 5, 10], None) == [1, 6, 15]
    assert recursive_summation([1, 6, 15, 20], None) == [1, 7, 21, 35]

def get_middle(_list):
    """Returns the middle value if lenght of list is odd or values if the length of the list is an even,"""
    l = len(_list)
    if (l % 2 == 0):
        m1 = int(l / 2)
        return m1 - 1, m1
    else:
        return int(l/2 - 0.5)


def test_get_middle_even():
    assert get_middle([1, 9, 9, 1]) == (1, 2)
    assert get_middle([1, 10, 1]) == 1
# Combinations and factorials

# Coefficients of binomial expansion
