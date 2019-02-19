import matplotlib.pyplot as plt
import numpy as np
from random import randint
from pivot import pivot, randRot2
from saw import polymer


def plotwalk(walk, style='-o', vertex=None, adjust=False):
    """Plot a walk and designated vertex"""

    x = [item[0] for item in walk]
    y = [item[1] for item in walk]

    colours = ['blue'] * len(x)
    default_size = plt.rcParams['lines.markersize'] ** 2
    sizes = [default_size] * len(x)

    if vertex is not None:
        colours[vertex] = 'red'
        sizes[vertex] = default_size * np.log10(len(x)) * 2

    if adjust:
        size = int(1.1 * walk.maxDist(np.inf))
    else:
        size = len(x)

    plt.clf()
    if '-' in style:
        plt.plot(x, y, zorder=0)
    if 'o' in style:
        plt.scatter(x, y, s=sizes, c=colours, zorder=1)
    plt.axis([-size, size, -size, size])
    plt.show()


def demo(steps, iterations, species='strict',
         wait=0.01, style='-o', **kwargs):
    """Run a demo of the pivot algorithm"""

    walk = polymer(steps, 2, species, **kwargs)

    plt.ion()
    for n in range(iterations):
        print('Iteration ', n)

        pivStep = randint(0, walk.steps - 1)
        plotwalk(walk, style, vertex=pivStep)
        plt.pause(wait)

        r = randRot2()
        new_walk = pivot(walk, pivStep, r)
        if new_walk != walk:
            walk = new_walk
            plotwalk(walk, style, vertex=pivStep)
            plt.pause(wait)
