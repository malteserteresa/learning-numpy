# Pascals triangle

def pascals_triangle(previous_row):
    """Prints next line of pascals triangle.

    Args:
        list : The next row following the given row

    """
    middle_idx = get_middle(previous_row)

    if (len(middle_idx) == 2):
        row = create_coefficients(previous_row[:middle_idx[1]])
        return row + [previous_row[middle_idx[1]] * 2] + row[::-1]
    else:
        row = create_coefficients(previous_row[:middle_idx[0]])
        middle = [row[-1] * 2]
        return row + middle + middle + row[::-1]


def test_even():
    assert pascals_triangle([1, 3, 3, 1]) == [1, 4, 6, 4, 1]


def test_odd():
    assert pascals_triangle([1, 4, 6, 4, 1]) == [1, 5, 10, 10, 5, 1]


def create_coefficients(split_row, first=None):
    """Returns the one half of the coefficients of pascals triangle for the next row.
    """
    if first == None:
        first = []

    # BASE CASE list = [1]
    if split_row == [1]:
        return split_row + first[::-1]
    else:
        first.append(split_row[-2] + split_row[-1])
        split_row = split_row[:-1]
        return create_coefficients(split_row, first)


def test_creates_half_of_next_row():
    assert create_coefficients([1, 3], None) == [1, 4]
    assert create_coefficients([1, 5, 10], None) == [1, 6, 15]
    assert create_coefficients([1, 6, 15, 20], None) == [1, 7, 21, 35]


def get_middle(_list):
    """Returns the middle value if length of list is odd or values if the length of the list is an even."""
    l = len(_list)

    if l % 2 == 0:
        m1 = int(l / 2)
        return [m1 - 1, m1]
    else:
        return [int(l / 2 - 0.5)]


def test_get_middle_even():
    assert get_middle([1, 9, 9, 1]) == [1, 2]
    assert get_middle([1, 10, 1]) == [1]

# Coefficients of binomial expansion
