from collections import Counter
import logging
from threading import Lock
from time import asctime

__author__ = 'http://code.activestate.com/recipes/52201/'

class Memoize:
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will only work on functions with non-mutable arguments
    """
    logger = logging.Logger(__name__)
    logger.addHandler(logging.StreamHandler())
    factor = 0.5    # when cache is full, reduce to this portion

    def __init__(self, fn, max_cache=100000):
        self.fn = fn
        self.max_cache = max_cache
        self.memo = {}
        self.logger.debug("Maximum cache size {0}.".format(self.max_cache))

    def __call__(self, arg):
        if self.memo.has_key(arg):
            result = self.memo[arg]
        elif len(self.memo) > self.max_cache:
            result = list()  # TODO: the result value should not be function-specific
            logging.debug("Cache limit reached, ignoring '{0}'.".format(arg))
        else:
            result = self.fn(arg)
            self.memo[arg] = result
        return result

    def _clean_cache(self):
        self.logger.debug(asctime() + " Reducing memory to {0} most common elements...".format(self.max_cache))
        cleaned = dict()
        for e in self.arg_counter.most_common(int(self.max_cache * self.factor)):
            if e[0] in self.memo.keys():
                cleaned[e[0]] = self.memo[e[0]]
        return cleaned
