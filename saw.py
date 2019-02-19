import numpy as np
from collections.abc import Sequence


def energy_strict(walk):
    """Energy for a strictly self-avoiding walk."""
    pass


def energy_weak(walk):
    """
    The energy function for a weakly self-avoiding walk with
    nearest-neighbour self-attraction.
    """

    try:
        repulsion = walk.kwargs['repulsion']
    except KeyError:
        repulsion = 0
    try:
        attraction = walk.kwargs['attraction']
    except KeyError:
        attraction = 0

    intersections = 0
    contacts = 0

    for i in range(len(walk)):
        for j in range(i + 1, len(walk)):
            dist = np.linalg.norm(walk[i] - walk[j], ord=1)
            if dist == 0:
                intersections += 1
            elif dist == 1:
                contacts += 1

    # The following is missing some kind of normalization
    return repulsion * intersections - attraction * contacts


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

        if species is None:
            self.species = 'custom'
            self.energy_fcn = kwargs.get('energy')
        else:
            self.species = species

        if self.species == 'weak':
            self.repulsion = kwargs.get('repulsion')
            self.attraction = kwargs.get('attraction')
            self.energy_fcn = energy_weak
        elif self.species == 'strict':
            self.energy_fcn = energy_strict
        elif self.species == 'simple':
            self.energy_fcn = lambda x: 0
        else:
            self.energy_fcn = kwargs.get('energy')

        super().__init__()

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        return self.path[i]

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
        return self.energy_fcn(self)

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
