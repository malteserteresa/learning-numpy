import matplotlib.pyplot as plt
import numpy as np
import plot_radians


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