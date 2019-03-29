import numpy as np
import matplotlib.pyplot as plt
import plot_radians

g = 9.81

def basic_f(y, v0):
    """ The equation for the trajectory of a ball can be solved to find the time it takes the ball to reach a certain height
        How long does it take for the ball to reach 0.2m with an initial velocity of 5m per se?
    """
    sqrt_term = np.sqrt((v0**2 - 2*g*y))
    return (v0 - sqrt_term, v0 + sqrt_term)


print(f"At t={basic_f(0.2, 5.0)[0]} s and {basic_f(0.2, 5.0)[1]} s, the height is 0.2 m")



def f(x, theta, v0, y):
    """ A ball is either kicked or thrown with a certain velocity at a certain angle. Calculate the height when it is 5m
        from it's original position
    """
    const = 1/(v0*2)

    return x*np.tan(theta) - const*(g*x**2/(np.cos(theta)*np.cos(theta))) + y


print(f"The height of the ball at 0.1 metres from the starting position is {round(f(0.1, 60 * (np.pi/180), 15, 1), 2)} m")



def plot(x, f, radians = False):
    """ A tool to help better understand the impact of various mathematical operations on the function
    """

    plt.plot(x, f)
    ax = plt.gca() # get current axis
    ax.grid(True)
   # ax.set_aspect(0.05) # something to do with size
    ax.axhline(0, color='black', lw=2)
    ax.axvline(0, color='black', lw=2)
    if(radians):
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(plot_radians.multiple_formatter()))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.abs(x.max())))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.abs(x.min()+1)))

    plt.show()


#x = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 20)
x = np.linspace(0, 2, 10)
plot(x, f(x, 60 * (np.pi/180), 15, 1), False)

