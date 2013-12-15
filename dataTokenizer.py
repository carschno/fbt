from nltk import PunktWordTokenizer, PunktSentenceTokenizer, clean_html
import pandas as pd
import logging
import itertools
from time import asctime
from collections import Counter

import joblib
import functools

from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

tokenizer = PunktWordTokenizer()
sentence_splitter = PunktSentenceTokenizer()


def tokenize(text, tagset):
    return pd.Series(Counter(filter(lambda x: x in tagset, map(str.lower, itertools.chain.from_iterable(
        map(tokenizer.tokenize, sentence_splitter.tokenize(text)))))))


def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def main():
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"
    outfile = "/home/carsten/facebook/cache/tokenizations"

    tagcache = "/home/carsten/facebook/cache/tags"
    logger.info("{0} Reading tags from '{1}...".format(asctime(), tagcache))
    tags = joblib.load(tagcache)

    data = read_zip(trainingzip, trainingfile, cols=["Id", "Title", "Body", "Tags"], index_col=0,
                          count=nrows).drop_duplicates(cols="Title")
    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)

    predictions = pd.merge(test, data, on=["Title", "Body"], how="left").drop_duplicates("Id")

    #predictions = pd.merge(test, data, on="Title", how="left", left_index=True, right_index=True)
    missing = predictions.index[predictions.Tags.isnull()]

    logger.info(asctime() + " Tokenizing {0} entries...".format(len(missing)))
    tokenize_ = functools.partial(tokenize, tagset=set(tags.index))

    tokenizations = pd.Series(index=predictions.Id[missing], dtype="O")
    tokenizations[predictions.Id[missing]] = (predictions.Title[missing] + " " + predictions.Body[missing].map(clean_html)).map(tokenize_)

    logger.info(asctime() + " Writing to file '{0}'".format(outfile))
    joblib.dump(tokenizations, outfile)
    logger.info(asctime()+ " Done.")

if __name__ == "__main__":
    nrows = None

    main()