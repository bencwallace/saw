import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer

from math import cos, sin, pi
from random import randint


def saw(steps, iterations, interactive=True):
    """Generate and plot a self-avoiding walk using the pivot algorithm

    Args:
        steps (int): The number of steps in the walk.
        iterations (int): The number of times to pivot the walk.
        interactive (bool, optional): Set to True to update the
            plot after each iteration, and set to False otherwise.
            Defaults to True.

    Returns:
        list: The generated self-avoiding walk

    """

    start = default_timer()

    walk = [np.array([i, 0]) for i in range(0, steps)]

    if interactive:
        plt.ion()
    plt.clf()
    plt.show()

    for n in range(0, iterations):
        if interactive:
            print("Iteration %d\n" % n)
        elif n % 100 == 0:
            print("Iteration %d\n" % n)

        pivWalk, pivStep = pivot(walk)
        if not checkIntersect(pivWalk, pivStep):
            walk = pivWalk

        x = [item[0] for item in walk]
        y = [item[1] for item in walk]
        if interactive:
            plt.clf()
            plt.plot(x, y, '-o')
            plt.axes().set_aspect('equal', 'datalim')
            plt.pause(0.5)

    plt.plot(x, y, '-o')
    plt.axes().set_aspect('equal', 'datalim')
    plt.pause(0.1)

    stop = default_timer()
    if not interactive:
        print('Run iterations: ', stop - start, 's')

    return walk


def pivot(walk):
    """Perform a single random pivot of a walk"""

    steps = len(walk)
    pivStep = randint(0, steps - 1)
    pivPoint = walk[pivStep]

    angle = randint(0, 3) * pi / 2
    cosAng = round(cos(angle))
    sinAng = round(sin(angle))
    rot = np.array([[cosAng, sinAng], [-sinAng, cosAng]])

    pivWalk = walk[:]
    for i in range(pivStep, steps):
        pivWalk[i] = (pivPoint +
                      np.dot(rot, np.transpose(pivWalk[i] - pivPoint)))

    return pivWalk, pivStep


def checkIntersect(walk, pivStep):
    """Check for self-intersections in a walk"""

    steps = len(walk)

    for i in range(0, pivStep):
        for j in range(pivStep + 1, steps):
            if (walk[i] == walk[j]).all():
                return True

    return False
