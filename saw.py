import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Sequence


def energy_strict():
    pass


def energy_mixed(walk, repulsion, attraction):
    """
    The energy function for a weakly self-avoiding walk with
    nearest-neighbour self-attraction. A combined implementation is more
    efficient.
    """

    intersections = 0
    contacts = 0

    for i in range(len(walk)):
        for j in range(i + 1, len(walk)):
            dist = np.linalg.norm(walk[i] - walk[j], ord=1)
            if dist == 0:
                intersections += 1
            if attraction != 0 and dist == 1:
                contacts += 1

    # The following is missing some kind of normalization
    return repulsion * intersections - attraction * contacts


def energy_weak(walk, repulsion):
    return energy_mixed(walk, repulsion, 0)


def energy_attract(walk, attraction):
    return energy_mixed(walk, 0, attraction)


class polymer(Sequence):
    """A linear polymer"""

    known_species = ['simple', 'strict', 'weak', 'attract', 'mixed']

    def __init__(self, steps, species='strict', **kwargs):
        if species not in polymer.known_species:
            self.species = 'custom'
            self.energy_fcn = kwargs.get('energy')
        else:
            self.species = species

        # Get repulsion and attraction
        if self.species == 'weak' or self.species == 'mixed':
            self.repulsion = kwargs.get('repulsion')
        if self.species == 'attract' or self.species == 'mixed':
            self.attraction = kwargs.get('attraction')

        # Initialize walk
        if type(steps) is int:
            self.steps = steps
            self.path = np.array([[i, 0] for i in range(self.steps)])
        elif type(steps) is list:
            self.steps = len(steps)
            self.path = steps

        super().__init__()

    def __getitem__(self, i):
        return self.path[i]

    def __len__(self):
        return self.steps

    def energy(self):
        if self.species == 'simple':
            return 0
        elif self.species == 'strict':
            return energy_strict(self)
        elif self.species == 'weak':
            return energy_weak(self)
        elif self.species == 'attract':
            return energy_attract(self)
        elif self.species == 'mixed':
            return self.repulsion * energy_weak(self) -\
                self.attraction * energy_attract(self)
        else:
            return self.energy_fcn(self)

    def display(self):
        for point in self.path:
            print(point)

    def dist(self, start=0, end=-1, ord=2):
        """Return the end-to-end distance of the walk"""

        return np.linalg.norm(self.path[end] - self.path[start], ord)

    def maxDist(self, ord=2):
        """Return the max distance between any two points of the walk"""

        m = 0
        for i in range(self.steps):
            for j in range(i + 1, self.steps):
                m = max(self.dist(i, j, ord), m)
        return m
