from collections import Counter, MutableMapping
import logging

__author__ = 'carsten'


class WeightedMap(MutableMapping):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    def __init__(self):
        self.vertices = dict()

    def __add__(self, other):
        self.vertices += other.vertices

    def __getitem__(self, item):
        return self.vertices[item]

    def __setitem__(self, key, value):
        assert type(value) == Counter
        self.vertices = value

    def __delitem__(self, key):
        del self.vertices[key]

    def __iter__(self):
        return self.vertices.__iter__()

    def __len__(self):
        return len(self.vertices)

    def add_vertice(self, source, target, weight=1):
        if source not in self.vertices.keys():
            self.logger.debug("Adding new vertice: '{0}'.".format(source))
            self.vertices[source] = Counter()
        self.logger.debug("Increasing weight from '{0}' to '{1}'.".format(source, target))
        self.vertices[source][target] += weight
        self.logger.debug("New weight: {0}".format(self.vertices[source][target]))

    def get_weight(self, source, target):
        if source in self.vertices.keys():
            return self.vertices[source][target]
        else:
            return 0

    def get_weights(self, source):
        if source in self.vertices.keys():
            return self.vertices[source]
        else:
            return Counter()

    def pruned_map(self, threshold=1):
        new_map = WeightedMap()
        for source in self.vertices:
            for target in self.vertices[source]:
                if self.get_weight(source, target) > threshold:
                    new_map.add_vertice(source, target, self.get_weight(source, target))
        return new_map

