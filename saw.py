import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Sequence
from functools import partial
from random import randint, random
from copy import deepcopy


class saw(Sequence):
    """Self-avoiding walk class"""

    def __init__(self, init, energy='strict', params=None):
        if energy == 'strict':
            self.energy = 'strict'
        elif energy == 'weak':
            self.energy = partial(wsaw_sa, attraction=0)
        else:
            self.energy = energy

        self.params = params

        if type(init) is int:
            self.steps = init
            self.w = np.array([[i, 0] for i in range(self.steps)])
        elif type(init) is list:
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

    def pivotStrict(self, pivStep, rot):
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
            # Checking for intersections can be optimized using hash tables
            for j in range(pivStep):
                # Might be a bit faster (especially if d is large) to check one
                # component at a time.
                # See also the speedup for nearest-neighbour walks in Stellman,
                # Froimowitz, and Gans (71)
                if (pivWalk[i] == pivWalk[j]).all():
                    return False

        self.w = pivWalk
        return True

    def pivotEnergy(self, pivStep, rot):
        """Pivot according to an energy function"""

        # First count intersections
        E = self.energy(self, *self.params)
        # for i in range(self.steps):
        #     for j in range(i + 1, self.steps):
        #         if (self.w[i] == self.w[j]).all():
        #             intersections += 1

        # Copy, pivot, and count new intersections
        # pivIntersections = 0
        pivWalk = deepcopy(self.w)
        pivPoint = pivWalk[pivStep]
        for i in range(pivStep + 1, self.steps):
            pivWalk[i] = (pivPoint +
                          np.dot(rot, np.transpose(pivWalk[i] - pivPoint)))
            # for j in range(pivStep):
            #     if (pivWalk[i] == pivWalk[j]).all():
            #         pivIntersections += 1
        pivE = self.energy(self, *self.params)

        if pivE <= E:
            self.w = pivWalk
            return True

        # ratio = np.exp(-self.strength * (pivE - E))
        ratio = np.exp(-(pivE - E))
        r = random()
        if r < ratio:
            self.w = pivWalk
            return True

        return False

    def pivot(self, pivStep, rot):
        if self.energy == 'strict':
            return self.pivotStrict(pivStep, rot)
        else:
            return self.pivotEnergy(pivStep, rot)

    def mix(self, iterations):
        for n in range(0, iterations):
            if n > 0 and n % 100 == 0:
                print("Iteration %d\n" % n)

            rot = randRot2()
            pivStep = randint(0, self.steps - 1)
            self.pivot(pivStep, rot)

    def dist(self, start=0, end=-1, ord=2):
        """Return the end-to-end distance of the walk"""

        return np.linalg.norm(self.w[end] - self.w[start], ord)

    def maxDist(self, ord=2):
        """Return the max distance between any two points of the walk"""

        m = 0
        for i in range(self.steps):
            for j in range(i + 1, self.steps):
                m = max(self.dist(i, j, ord), m)
        return m


def wsaw_sa(walk, repulsion, attraction):
    """
    The energy function for a weakly self-avoiding walk with
    nearest-neighbour self-attraction
    """

    intersections = 0
    contacts = 0

    for i in range(walk.steps):
        for j in range(i + 1, walk.steps):
            dist = np.linalg.norm(walk[i] - walk[j], ord=1)
            if dist == 0:
                intersections += 1
            if attraction != 0 and dist == 1:
                contacts += 1

    return repulsion * intersections - attraction * contacts


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
    # plt.axes().set_aspect('equal', 'datalim')
    # size = max(max(plt.axis()), walk.maxDist())
    size = walk.maxDist(np.inf) + 10
    plt.axis([-size, size, -size, size])
    plt.pause(0.2)


def demo(steps, iterations, strength=np.inf, style='-o'):
    """Run a demo of the pivot algorithm"""

    plt.ion()

    walk = saw(steps, strength)

    for n in range(iterations):
        print('Iteration ', n)
        walk.mix(1)
        plotwalk(walk, style)
    return None
