import pandas as pd
import csv
import logging
from nltk import word_tokenize, wordpunct_tokenize, sent_tokenize, clean_html
from nltk.corpus import stopwords
from time import asctime
from string import punctuation
from collections import Counter

import joblib

from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

__author__ = 'carsten'


def match_tags(tokens, tag_series, max_tags=5):
    """

    @param text:
    @type text: pd.Series
    @param tag_series:
    @type tag_series: pd.Series
    @param max_tags:
    @type max_tags: int
    @return:
    @rtype: pd.Series
    """
    tags = (tokens * tag_series[tokens.index]).fillna(0)
    tags.sort(ascending=False)
    tags = tags[:max_tags]
    tags = list(tags.drop(tags.index[tags < tags.mean()]).index)
    return tags


def tokenize(text):
    # TODO optimize for speed
    # TODO optimze tokenizer to match tags, include multiword expressions; iterate over tags and find in text? remove hyphens?
    tokens = list()
    for sent in sent_tokenize(text):
        tokens.extend(filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
                             word_tokenize(sent.lower()) + wordpunct_tokenize(text.lower())))
    return pd.Series(Counter(tokens))


def main():
    data = read_zip(trainingzip, trainingfile, cols=["Id", "Title", "Tags"], index_col=0, count=nrows).drop_duplicates(
        cols="Title")

    logger.info(asctime() + " Reading tag counts from {0}...".format(tagcache))
    tags = joblib.load(tagcache)

    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").sort(columns=["Id"]).drop_duplicates("Id")

    missing = predictions.index[predictions.Tags.isnull()]
    logger.info("{0} Computing {1} missing tags using tokens in title and in body.".format(asctime(), len(missing)))
    for i in missing:
        predictions.Tags[i] = " ".join(
            match_tags(tokenize(predictions.Title[i] + " " + clean_html(predictions.Body[i])), tags))

    outfile = "/tmp/predictions_{0}documents.csv".format(nrows)
    #outfile = "/home/carsten/facebook/predictions_{0}documents.csv".format(nrows)

    logger.info(asctime() + " Writing predictions to '{0}'...".format(outfile))
    predictions.to_csv(outfile, index=False, cols=("Id", "Tags"), quoting=csv.QUOTE_ALL)
    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"

    columns = ["Id", "Title", "Tags"]
    tagcache = "/home/carsten/facebook/cache/tags"

    nrows = 1000

    main()
