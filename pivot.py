import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Sequence
# from timeit import default_timer
from random import randint
from copy import deepcopy


class saw(Sequence):
    """Self-avoiding walk class"""

    def __init__(self, init):
        if type(init) == int:
            self.steps = init
            self.w = np.array([[i, 0] for i in range(self.steps)])
        elif type(init) == list:
            self.steps = len(init)
            self.w = init
        super().__init__()

    def __getitem__(self, i):
        return self.w[i]

    def __len__(self):
        return self.steps

    def display(self):
        for point in self.w:
            print(point)

    def pivotOnce(self, pivStep, rot):
        """Attempt to pivot the walk in a self-avoiding manner

        Args:
            pivStep (int): the step about which to pivot
            rot (numpy.array): a rotation matrix

        Returns:
            bool: True if pivot succeeds, False otherwise
        """

        # Copy walk into pivWalk and attempt to pivot steps past pivStep
        pivWalk = deepcopy(self.w)
        pivPoint = pivWalk[pivStep]
        for i in range(pivStep + 1, self.steps):
            pivWalk[i] = (pivPoint +
                          np.dot(rot, np.transpose(pivWalk[i] - pivPoint)))
            # Check for intersection resulting from pivot. Intersections can
            # only occur between steps before and after pivStep.
            for j in range(pivStep):
                if (pivWalk[i] == pivWalk[j]).all():
                    return False

        self.w = pivWalk
        return True

    def pivot(self, iterations):
        for n in range(0, iterations):
            if n > 0 and n % 100 == 0:
                print("Iteration %d\n" % n)

            rot = randRot2()
            pivStep = randint(0, self.steps - 1)
            self.pivotOnce(pivStep, rot)

    def dist(self, start=0, end=-1):
        """Return the end-to-end distance of the walk"""

        return np.linalg.norm(self.w[end] - self.w[start])

    def maxDist(self):
        """Return the max distance between any two points of the walk"""

        m = 0
        for i in range(self.steps):
            for j in range(i + 1, self.steps):
                m = max(self.dist(i, j), m)
        return m


def randRot2():
    """Return a random 2 by 2 rotation matrix that is not the identity"""

    rot = np.empty([2, 2])

    rot[0, 1] = randint(-1, 1)
    rot[0, 0] = abs(rot[0, 1]) - 1
    rot[1, 0] = -rot[0, 1]
    rot[1, 1] = rot[0, 0]

    return rot


def plotwalk(walk, style='-o'):
    """Plot a walk"""

    x = [item[0] for item in walk]
    y = [item[1] for item in walk]
    plt.clf()
    plt.plot(x, y, style)
    plt.axes().set_aspect('equal', 'datalim')
    plt.pause(0.2)


def demo(steps, iterations):
    """Run a demo of the pivot algorithm"""

    plt.ion()

    walk = saw(steps)

    for n in range(iterations):
        print('Iteration ', n)
        walk.pivot(1)
        plotwalk(walk)
    return None
