__author__ = 'carsten'

import unittest
import logging

from WeightedMap import WeightedMap


class MyTestCase(unittest.TestCase):
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)


    def test_add(self):
        add_map = WeightedMap()
        add_map.add_vertice("A", "B")
        self.assertIn("A", add_map.vertices.keys())
        self.assertNotIn("B", add_map.vertices.keys())
        self.assertEquals(1, add_map.vertices["A"]["B"])

        add_map.add_vertice("A", "B")
        self.assertIn("A", add_map.vertices.keys())
        self.assertNotIn("B", add_map.vertices.keys())
        self.assertEquals(2, add_map.vertices["A"]["B"])

    def test_weighted_add(self):
        add_map = WeightedMap()
        add_map.add_vertice("A", "B", 2)
        self.assertEquals(2, add_map.vertices["A"]["B"])

    def test_get(self):
        get_map = WeightedMap()
        get_map.add_vertice("A", "B")
        self.assertEquals(1, get_map.get_weight("A", "B"))
        self.assertEquals(0, get_map.get_weight("A", "C"))
        self.assertEquals(0, get_map.get_weight("B", "A"))

        get_map.add_vertice("A", "B")
        self.assertEquals(2, get_map.get_weight("A", "B"))

    def test_get_weights(self):
        get_map = WeightedMap()
        get_map.add_vertice("A", "B")
        get_map.add_vertice("A", "D")
        weights_a = get_map.get_weights("A")
        self.assertListEqual(["B", "D"], weights_a.keys())

    def test_len(self):
        weighted_map = WeightedMap()
        self.assertEquals(0, len(weighted_map))

        weighted_map.add_vertice("A", "B")
        self.assertEquals(1, len(weighted_map))

        weighted_map.add_vertice("A", "C")
        self.assertEquals(1, len(weighted_map))

        weighted_map.add_vertice("B", "C")
        self.assertEquals(2, len(weighted_map))

    def test_prune(self):
        weighted_map = WeightedMap()
        weighted_map.add_vertice("A", "B")
        weighted_map.add_vertice("A", "C", 2)
        weighted_map.add_vertice("B", "C", 2)

        pruned_map = weighted_map.pruned_map()
        self.assertListEqual(["A","B"], pruned_map.vertices.keys())
        self.assertEquals(0, pruned_map.get_weight("A", "B"))
        self.assertEquals(2, pruned_map.get_weight("A", "C"))
        self.assertEquals(2, pruned_map.get_weight("B", "C"))

if __name__ == '__main__':
    unittest.main()
