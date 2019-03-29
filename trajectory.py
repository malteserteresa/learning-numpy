import numpy as np
import matplotlib.pyplot as plt
import plot_radians

v0 = 5.0
g = 9.81
y = 0.2


def basic_f(y):
    """ The equation for the trajectory of a ball can be solved to find the time it takes the ball to reach a certain height
        How long does it take for the ball to reach 0.2m with an initial velocity of 5m per se?
    """
    sqrt_term = np.sqrt((v0**2 - 2*g*y))
    return (v0 - sqrt_term, v0 + sqrt_term)


print(f"At t={basic_f(y)[0]} s and {basic_f(y)[1]} s, the height is 0.2 m")



def f(x):
    """ A ball is either kicked or thrown with a certain velocity at a certain angle. Calculate the height when it is 5m
        from it's original position
    """
    const = 1/(v0*2)

    return const


def test_constant():
    assert 1/(2*v0**2) == 0.02



def plot(x, f):
    """ A tool to help better understand the impact of various mathematical operations on the graph

    """

    plt.plot(x, f)
    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect(1.0)
    ax.axhline(0, color='black', lw=2)
    ax.axvline(0, color='black', lw=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(plot_radians.multiple_formatter()))
    plt.show()


x = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 20)
plot(x, x*np.tan(x))
