import pandas as pd
from Utils import read_zip
from collections import Counter
from time import asctime
import joblib
import logging

__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def count_tags(tag_strings):
    """

    @param tag_strings:
    @type tag_strings: pd.Series
    @return:
    @rtype: pd.Series
    """
    c = Counter()
    for tag_string in tag_strings:
        if len(c) % 1000 == 0:
            logger.debug(asctime() + " {0} unique tags found.".format(len(c)))
        c.update(tag_string.split())
    return pd.Series(c)

def main():
    data = read_zip(trainingzip, trainingfile, cols=["Tags"], count=10000)
    joblib.dump(count_tags(data.Tags), tagcache)

if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    #tagcache = "/home/carsten/facebook/cache/tags"
    tagcache = "/tmp/tags"

    main()

