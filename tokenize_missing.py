from nltk import PunktSentenceTokenizer, WordPunctTokenizer, clean_html
import pandas as pd
import logging
import itertools
from time import asctime
from collections import Counter
import functools

import joblib

from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# adapt output names when changing tokenizer!
#tokenizer = PunktWordTokenizer()
sentence_splitter = PunktSentenceTokenizer()
tokenizer = WordPunctTokenizer()


def tokenize(text, tagset, splitchar="-"):
    """
    Compute a bag of words of the given text, retaining only the tokens contained in the tagset.
    Performs sentence splitting and tokenization.
    @param text: A string to tokenize
    @type text: str
    @param tagset: A set containing the tokens to retain in result
    @type tagset: set
    @return: A Series mapping tokens to frequencies
    @rtype: pd.Series
    """
    return pd.Series(Counter(filter(lambda x: x in tagset, map(str.lower, itertools.chain.from_iterable(
        map(tokenizer.tokenize, sentence_splitter.tokenize(text)))))))


def chunks(l, n):
    """
    Return a generator dividing the list l into chunks of size n.
    @param l:
    @param n:
    @return:
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def get_tagset(s, splitchar="-"):
    """
    Generate a tagset from the given series index while splitting tags from multiple tokens
    @param s:
    @param splitchar: character to detect multi-token indizes
    @return:
    @rtype: set
    """
    logger.info(asctime() + " Generating tagset from {0} indizes.".format(len(s)))
    tagset = set(s.index)

    logger.info(asctime() + " Finding multi-word expressions in tagset.")
    mwes = set()
    for tag in itertools.ifilter(lambda x: splitchar in x, tagset):
        mwes.update(str.split(tag.replace(splitchar, " ")))
    logger.info(asctime() + " Adding {0} tokens from multi-word expressions to tagset.".format(len(mwes)))
    return tagset | mwes


def main():
    """
    Perform tokenizations for each document in test data that does not have a duplicate title in training data, using
    the globally defined tokenizer.
    The results are written to disk in chunks with an additional index file.
    """

    tokenizationsindex = dict()
    logger.info("{0} Reading tags from '{1}...".format(asctime(), tagcache))
    tags = joblib.load(tagcache)
    data = read_zip(trainingzip, trainingfile, cols=["Id", "Title", "Tags"], index_col=0,
                    count=nrows).drop_duplicates(cols="Title")
    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)
    logger.info(asctime() + " Merging training set and test set...")
    predictions = pd.merge(test, data, on="Title", how="left").drop_duplicates("Id")
    missing = predictions.index[predictions.Tags.isnull()]
    logger.info(asctime() + " Tokenizing {0} entries from {1} to {2}...".format(
        len(missing), predictions.Id[missing[0]], predictions.Id[missing[-1]]))
    tokenize_ = functools.partial(tokenize, tagset=get_tagset(tags))

    for indizes in chunks(missing, 50000):
        logger.info(asctime() + " Tokenizing indizes between {0} and {1}...".format(predictions.Id[indizes[0]],
                                                                                    predictions.Id[indizes[-1]]))
        outfile = tokenizationsfile + str(indizes[0])
        tokenizationsindex[predictions.Id[indizes[0]]] = outfile
        tokenizations = pd.Series(index=predictions.Id[indizes], dtype="O")
        tokenizations[predictions.Id[indizes]] = (
            predictions.Title[indizes] + " " + predictions.Body[indizes].map(clean_html)).map(tokenize_)

        logger.info(
            asctime() + " Writing tokenizations for indizes between {0} and {1} to file '{2}'.".format(
                predictions.Id[indizes[0]], predictions.Id[indizes[-1]], outfile))
        joblib.dump(tokenizations, outfile)

    logger.info(asctime() + " Writing tokenization index  to '{0}'.".format(tokenizationsindexfile))
    joblib.dump(tokenizationsindex, tokenizationsindexfile)

    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"
    tagcache = "/home/carsten/facebook/cache/tags"

    nrows = 10000
    # adapt output names to tokenizer!
    tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex_wordpunct" + str(nrows)
    tokenizationsfile = "/home/carsten/facebook/cache/tokenizations_wordpunct" + str(nrows)

    main()