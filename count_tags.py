import pandas as pd
import numpy as np
from collections import Counter
from time import asctime
import logging

import joblib

from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def count_tags(tag_strings, sort=True):
    """

    @param tag_strings:
    @type tag_strings: pd.Series
    @return:
    @rtype: pd.Series
    """
    c = Counter()
    for tag_string in tag_strings:
        c.update(tag_string.split())
    tags = pd.Series(c, dtype=np.int)
    if sort:
        tags.sort(ascending=False)
    logger.info(tags.describe())
    return tags


def main():
    """
    Count all tags in training data and write to file.
    """
    data = read_zip(trainingzip, trainingfile, cols=["Tags"], count=None)
    logger.info("{1} Writing cache to {0}.".format(tagcache, asctime()))
    joblib.dump(count_tags(data.Tags), tagcache)
    logger.info("{0} Done.".format(asctime()))


if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    tagcache = "/home/carsten/facebook/cache/tags"

    main()

