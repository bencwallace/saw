import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
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

        walk = pivot(walk)

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
        print('Run time: ', stop - start, 's')

    return walk


def pivot(walk):
    """Attempt to perform a single random pivot of a self-avoiding walk"""

    steps = len(walk)
    pivStep = randint(0, steps - 1)
    pivPoint = walk[pivStep]

    rot = randRot2()

    pivWalk = walk[:]
    for i in range(pivStep + 1, steps):
        pivWalk[i] = (pivPoint +
                      np.dot(rot, np.transpose(pivWalk[i] - pivPoint)))
        for j in range(pivStep):
            if (pivWalk[i] == pivWalk[j]).all():
                return walk

    return pivWalk


def randRot2():
    """Return a random 2 by 2 rotation matrix that is not the identity"""
    rot = np.empty([2, 2])

    rot[0, 1] = randint(-1, 1)
    rot[0, 0] = abs(rot[0, 1]) - 1
    rot[1, 0] = -rot[0, 1]
    rot[1, 1] = rot[0, 0]

    return rot
