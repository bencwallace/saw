import matplotlib.pyplot as plt
import numpy as np
from pivot import pivot_energy, pivot_strict
from saw import polymer


def plotwalk(walk, style='-o', vertex=None):
    """Plot a walk and designated vertex"""

    x = [item[0] for item in walk]
    y = [item[1] for item in walk]
    # colours = ['blue'] * len(x)
    default_size = plt.rcParams['lines.markersize'] ** 2
    sizes = [default_size] * len(x)
    if vertex is not None:
        # colours[vertex] = 'red'
        sizes[vertex] = default_size * np.log10(len(x)) * 2

    plt.clf()
    # if style != 'o':
    #     plt.plot(x, y, style)
    plt.scatter(x, y, s=sizes)
    size = int(1.1 * walk.maxDist(np.inf))
    plt.axis([-size, size, -size, size])
    plt.show()


def demo(steps, iterations, energy='strict', style='-o', **kwargs):
    """Run a demo of the pivot algorithm"""

    plt.ion()

    walk = polymer(steps, energy, **kwargs)

    for n in range(iterations):
        print('Iteration ', n)

        pivStep = randint(0, walk.steps - 1)
        plotwalk(walk, style, vertex=pivStep)

        plt.pause(0.5)

        r = randRot2()
        walk.pivot(pivStep, r)
        plotwalk(walk, style)

        plt.pause(0.2)
    return None


# def demo(steps, iterations, energy='srw', style='-o', **kwargs):
#     """Run a demo of the pivot algorithm"""

#     plt.ion()

#     walk = saw(steps, energy, **kwargs)

#     for n in range(iterations):
#         print('Iteration ', n)
#         walk.mix(1)
#         plotwalk(walk, style)
#         plt.pause(0.2)
#     return None
