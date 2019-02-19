from copy import deepcopy
import numpy as np
from random import random, randint
from saw import polymer


def randRot2():
    """Return a random 2 by 2 rotation matrix that is not the identity"""

    rot = np.empty([2, 2])

    rot[0, 1] = randint(-1, 1)
    rot[0, 0] = abs(rot[0, 1]) - 1
    rot[1, 0] = -rot[0, 1]
    rot[1, 1] = rot[0, 0]

    return rot


def pivot_strict(walk, step, rotation):
    """Attempt to pivot the walk in a self-avoiding manner"""

    # Copy walk into pivWalk and attempt to pivot steps past step
    newpath = deepcopy(walk.path)
    pivPoint = newpath[step]
    for i in range(step + 1, len(newpath)):
        diff = newpath[i] - pivPoint
        newpath[i] = (pivPoint + np.dot(rotation, np.transpose(diff)))

        # Possible ways to optimize: better search/hash tables, check one
        # component  at a time (large d), Stellman, Froimowitz, and Gans (71)
        if newpath[i] in walk:
            return walk

    return polymer(newpath, walk.dimension, walk.species, **walk.kwargs)


# Broken
def pivot_energy(walk, pivStep, rot):
    """Pivot according to an energy function"""

    pivWalk = deepcopy(walk.w)
    pivPoint = pivWalk[pivStep]
    for i in range(pivStep + 1, walk.steps):
        pivWalk[i] = (pivPoint +
                      np.dot(rot, np.transpose(pivWalk[i] - pivPoint)))
    E = walk.species(walk)
    pivE = walk.species(pivWalk)

    if pivE <= E:
        walk.w = pivWalk
        return True

    ratio = np.exp(-(pivE - E))
    r = random()
    if r < ratio:
        walk.w = pivWalk
        return True

    return False


def pivot(walk, pivStep, rot):
    if walk.species == 'strict':
        return pivot_strict(walk, pivStep, rot)
    else:
        return pivot_energy(walk, pivStep, rot)

# def mix(walk, iterations):
#     for n in range(0, iterations):
#         if n > 0 and n % 100 == 0:
#             print("Iteration %d\n" % n)

#         rot = randRot2()
#         pivStep = randint(0, walk.steps - 1)
#         walk.pivot(pivStep, rot)
