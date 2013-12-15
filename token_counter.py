import pandas as pd
import csv
import logging
from nltk import clean_html
from time import asctime
from collections import Counter
import joblib

from Utils import read_zip
import dataTokenizer

__author__ = 'carsten'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


__author__ = 'carsten'

def find_tags(token_series, tags, mwes=list(), max_tags=5):
    """

    @param text: a text
    @param mwes: a list of multi-word (multi-token) expressions to be searched in the text
    @type mwes: list
    @param mwe_separator: separator used in MWEs ("visual-studio-2010")
    @param max_tags: maximum number or tags assigned to text
    @return:
    """
    #tokens = dataTokenizer.tokenize(text, tags.index)
    #token_series = pd.Series(Counter(tokens))   # TODO: normalize token and tag frequencies
    #counts = pd.Series(0, index=set(tokens))


    # TODO: count MWEs
    #logger.debug("{0} Searching for {1} multi-word expressions...".format(asctime(), len(mwes)))
    #counts[mwes].map(text.lower().count)
    #for tag in tags.index:
    #    counts[tag] = tags[tag] * text.count(tag) + tags[tag] * text.count(tag.replace("-", " "))

    counts = (pd.Series(token_series) * tags[token_series.keys()]).dropna()

    counts.sort(ascending=False)
    counts = counts[counts.notnull()][:max_tags]
    result = list(counts.drop(counts.index[counts < counts.mean()]).index)
    return result


def mwe(s, char="-"):
    """
    Find all multi-word tags in s' index, using char as a separator. E.g.: visual-studio-2010
    @param s:
    @type s: pd.Series
    @param char:
    @type char: str
    @return:
    @rtype: list
    """
    mwes = list()
    for tag in filter(lambda x: char in x, list(s.index)):
        mwes.append(tag.replace(char, " "))
    return mwes


def main():
    data = read_zip(trainingzip, trainingfile, cols=["Id", "Title", "Tags"], index_col=0, count=nrows).drop_duplicates(
        cols="Title")

    logger.info(asctime() + " Reading tag counts from '{0}'...".format(tagcache))
    tags = joblib.load(tagcache, mmap_mode="r")

    logger.info(asctime() + " Loading tokenizations from '{0}'...".format(tokenizationfile))
    tokenizations = joblib.load(tokenizationfile, mmap_mode="r")

    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").drop_duplicates("Id")
    #predictions = pd.merge(test, data, on="Title", how="left", left_index=True, right_index=True)

    missing = predictions.index[predictions.Tags.isnull()]
    logger.info("{0} Computing {1} missing tags using tokens in title and in body.".format(asctime(), len(missing)))

    #predictions[missing] = (predictions.Title[missing].map(str.lower) + " " + predictions.Body[missing].map(clean_html).map(str.lower)).map(find_tags)

    #predictions[missing] = (predictions.Title[missing] + " " + predictions[missing].map(clean_html)).map(tokenize)
    counter = 0
    for i in missing:
        counter += 1
        if counter % 1000 == 0:
            logging.info("{0} {1} of {2} done.".format(asctime(), counter, len(missing)))
        #predictions.Tags[i] = " ".join(find_tags(predictions.Title[i] + " " + clean_html(predictions.Body[i])))
        #print tokenizations[predictions.Id[i]]
        if predictions.Id[i] in tokenizations.index:
            predictions.Tags[i] = " ".join(find_tags(tokenizations[predictions.Id[i]], tags))
        else:
            logger.warn(asctime() + " No tokenization found for {0}.".format(predictions.Id[i]))

    outfile = "/home/carsten/facebook/predictions_{0}documents.csv".format(nrows)


    logger.info(asctime() + " Writing predictions to '{0}'...".format(outfile))
    predictions.to_csv(outfile, index=False, cols=["Id", "Tags"], quoting=csv.QUOTE_ALL)
    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"

    columns = ["Id", "Title", "Tags"]
    tagcache = "/home/carsten/facebook/cache/tags"
    tokenizationfile = "/home/carsten/facebook/cache/tokenizations"

    nrows = 10000

    main()

