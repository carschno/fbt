import pandas as pd
import csv
import logging
from time import asctime
import joblib

from Utils import read_zip

__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def find_tags(token_series, tags, mwes=list(), max_tags=5):
    """

    @param token_series: a series representing a document's bag of words
    @type token_series: pd.Series
    @param mwes: a list of multi-word (multi-token) expressions to be searched in the text
    @type mwes: list
    @param max_tags: maximum number or tags assigned to text
    @type max_tags: 5
    @return: a list of tags assigned to the document
    @rtype: list
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

    logger.info(asctime() + " Loading tokenizations index from '{0}'...".format(tokenizationsindexfile))
    tokenizationindex = joblib.load(tokenizationsindexfile)

    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").drop_duplicates("Id")

    missing = predictions.index[predictions.Tags.isnull()]
    logger.info("{0} Computing {1} missing tags between {2} and {3}...".format(asctime(), len(missing),
                                                                               predictions.Id[missing[0]],
                                                                               predictions.Id[missing[-1]]))

    #predictions[missing] = (predictions.Title[missing].map(str.lower) + " " + predictions.Body[missing].map(clean_html).map(str.lower)).map(find_tags)

    #predictions[missing] = (predictions.Title[missing] + " " + predictions[missing].map(clean_html)).map(tokenize)

    tokenizations = pd.Series()
    for i in missing:
        #predictions.Tags[i] = " ".join(find_tags(predictions.Title[i] + " " + clean_html(predictions.Body[i])))
        #print tokenizations[predictions.Id[i]]
        if predictions.Id[i] not in tokenizations.index:
            logger.info(asctime() + " Loading indizes for {0} from '{1}'".format(predictions.Id[i],
                                                                                 tokenizationindex[predictions.Id[i]]))
            tokenizations = joblib.load(tokenizationindex[predictions.Id[i]])
        predictions.Tags[i] = " ".join(find_tags(tokenizations[predictions.Id[i]], tags))

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
    #tokenizationfile = "/home/carsten/facebook/cache/tokenizations"
    tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex"
    nrows = 100000

    main()

