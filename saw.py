import numpy as np
from collections.abc import Sequence


def energy_mixed(path, repulsion, attraction):
    """
    The energy function for a weakly self-avoiding walk with
    nearest-neighbour self-attraction.
    """

    intersections = 0
    contacts = 0

    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            dist = np.linalg.norm(path[i] - path[j], ord=1)
            if dist == 0:
                intersections += 1
            elif dist == 1:
                contacts += 1

    # The following is missing some kind of normalization
    return repulsion * intersections - attraction * contacts


def energy_weak(path, repulsion):
    """Energy for a weakly self-avoiding walk"""

    return energy_mixed(path, repulsion, 0)


def energy_strict(path):
    """Energy for a strictly self-avoiding walk."""

    # The number 1000 is chosen for use with Metropolis-Hastings,
    # where np.exp(-1000) == 0
    return 1000 * int(energy_weak(path, 1) > 0)


def energy_attract(path, attraction):
    """Energy for a walk with self-attraction."""

    return energy_mixed(path, 0, attraction)


class polymer(Sequence):
    """
    A linear polymer model.

    Attributes:
        known_species (list): "Known" polymer series.
        dimension (int): Dimension of ambient space.
        steps (int): Number of steps in walk.
        species (str): Species of walk.
        path (list): Polymer coordinates.
        energy_fcn (function): Custom energy function.
        repulsion (float): Repulsion parameter.
        attraction (float): Attraction parameter.

    Methods:
        energy: Return total energy of polymer.
        dist: Return end-to-end distance between parts of the polymer.
        maxDist: Returns the maximal end-to-end distance of the polymer.
    """

    known_species = ['simple', 'strict', 'weak', 'attract', 'mixed']

    def __init__(self, steps, dimension=2, species='strict', **kwargs):
        self.dimension = dimension
        self.kwargs = kwargs

        if type(steps) is int:
            self.steps = steps

            e1 = np.zeros(dimension)
            e1[0] = 1
            self.path = [i * e1 for i in range(self.steps)]
        else:
            self.steps = len(steps)
            self.path = steps

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

        super().__init__()

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        return polymer(self.path[i], self.dimension, self.species,
                       **self.kwargs)

    def __contains__(self, x):
        for item in self.path:
            if (item == x).all():
                return True
        return False

    def __neg__(self):
        neg_path = [-item for item in self.path]
        return polymer(neg_path, self.dimension, self.species,
                       **self.kwargs)

    def energy(self):
        if self.species == 'simple':
            return 0
        elif self.species == 'strict':
            return energy_strict(self.path)
        elif self.species == 'weak':
            return energy_weak(self.path)
        elif self.species == 'attract':
            return energy_attract(self.path)
        elif self.species == 'mixed':
            return energy_mixed(self.path)
        else:
            return self.energy_fcn(self.path)

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
