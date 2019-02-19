from copy import deepcopy
import numpy as np
from random import random, randint
from saw import polymer


def randRot2():
    """Return a random 2 by 2 lattice rotation besides the identity"""

    rot = np.empty([2, 2])

    rot[0, 1] = randint(-1, 1)
    rot[0, 0] = abs(rot[0, 1]) - 1
    rot[1, 0] = -rot[0, 1]
    rot[1, 1] = rot[0, 0]

    return rot


def pivot_strict(walk, step, rotation):
    """Attempt to pivot the walk in a self-avoiding manner"""

    newpath = deepcopy(walk.path)
    pivPoint = newpath[step]

    for i in range(step + 1, len(newpath)):
        diff = newpath[i] - pivPoint
        newpath[i] = (pivPoint + np.dot(rotation, np.transpose(diff)))
        if newpath[i] in walk:
            return walk
    new_walk = polymer(newpath, walk.dimension, walk.species, **walk.kwargs)

    return new_walk


def pivot_energy(walk, step, rotation):
    """Pivot according to an energy function"""

    newpath = deepcopy(walk.path)
    pivPoint = newpath[step]

    for i in range(step + 1, len(newpath)):
        diff = newpath[i] - pivPoint
        newpath[i] = pivPoint + np.dot(rotation, np.transpose(diff))
    new_walk = polymer(newpath, walk.dimension, walk.species, **walk.kwargs)

    E = walk.energy()
    pivE = new_walk.energy()
    ratio = np.exp(-(pivE - E))

    r = random()
    if ratio >= r:
        return new_walk
    else:
        return walk


def pivot(walk, step, rot):
    """Pivot a walk"""

    if walk.species == 'strict':
        return pivot_strict(walk, step, rot)
    else:
        return pivot_energy(walk, step, rot)


def mix(walk, iterations):
    new_walk = deepcopy(walk)

    for n in range(0, iterations):
        if n > 0 and n % 100 == 0:
            print("Iteration %d\n" % n)

        rot = randRot2()
        step = randint(0, walk.steps - 1)
        new_walk = pivot(new_walk, step, rot)

    return new_walk
