import pandas as pd
import csv
import logging
from time import asctime
import itertools
from collections import Counter
import joblib
from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def find_tags(document, tags, multitoken_i, max_tags=5):
    """
    Compute most likely tags for the given bag of words in token_series.
    The score for each tag comprises two factors: the frequency of a tag in the document and its frequency in the
     overall tags series.
    Scores for multi-token tags are added through find_mwe_tags.
    From all the scores, the n (max_tags) highest scoring ones are selected from which those that lie above average
     are returned eventually.

    @param document: a series representing a document bag of words
    @type document: pd.Series
    @param tags: all available tags and their frequencies
    @type tags: pd.Series
    @param multitoken_i: a map from tokens to mwe tags containing them
    @type multitoken_i: dict
    @param max_tags: maximum number or tags assigned to text
    @type max_tags: 5
    @return: a list of tags assigned to the document
    @rtype: list
    """

    scores = (document * tags[document.keys()]).dropna()    # no normalization done here
    scores = pd.concat([scores, find_mwe_tags(document, tags, multitoken_i)])
    scores.sort(ascending=False)
    scores = scores[scores.notnull()][:max_tags]
    result = list(scores.drop(scores.index[scores < scores.mean()]).index)
    return result


def find_mwe_tags(document, tags, multitoken_i, char="-"):
    """
    Find tags comprising multiple tokens through an inverted index "tokens -> multi-token-tags" in mwe_i.
    This method is relatively slow because it iterates over the complete inverted index for each document.

    @param document: a bag of words represeting a document (tokenized)
    @type document: pd.Series
    @param tags: all allowable tags with frequencies
    @type tags: pd.Series
    @param multitoken_i: an inverted index mapping from each token to a list of MWE tags containing that token
    @type multitoken_i: dict
    @return: a Series of scores for multi-token tags
    @rtype: pd.Series
    """
    multitoken_scores = dict()
    # iterate over all document tokens that occur in mwe index
    for token in itertools.ifilter(lambda x: x in multitoken_i, document.index):
        # iterate over all mwe tags that contain token
        for tag in multitoken_i[token]:
            if tag not in multitoken_scores and all(subtag in document.index for subtag in tag.split(char)):
                multitoken_scores[tag] = tags[tag] * (sum(document[tag.split(char)]) / len(tag.split(char)))
    return pd.Series(multitoken_scores).dropna()


def multitoken_index(tags, char="-"):
    """
    Generate an inverted index for tags comprising multiple tokens (e.g. 'visual-studio').
    Each token maps to a list of multi-token tags containing this token: {'visual': [visual-studio, ...], 'studio': [...]}

    @param tags: a series of tags in index
    @type tags: pd.Series
    @param char:
    @return:
    """
    index = dict()
    for tag in itertools.ifilter(lambda x: char in x, tags.index):
        for i in tag.split(char):
            if i not in index:
                index[i] = list()
            index[i].append(tag)
    return index


def main():
    """
    Main function. Read the test and training data, and tokenization. Apply find_tags for test documents that do not have a
    duplicate title in the training data to compute tags. Finally, write all results to file.
    @return: None
    """
    data = read_zip(trainingzip, trainingfile, cols=["Id", "Title", "Tags"], index_col=0, count=nrows).drop_duplicates(
        cols="Title", take_last=True)   # TODO: take_last=True
    test = read_zip(testzip, testfile, cols=["Id", "Title", "Body"], count=nrows)

    logger.info(asctime() + " Reading tag counts from '{0}'...".format(tagcache))
    tags = joblib.load(tagcache, mmap_mode="r") # no normalization done here
    multitoken_i = multitoken_index(tags)

    logger.info(asctime() + " Loading punkt_tokenizations index from '{0}'...".format(punkt_tokenizationsindexfile))
    punkt_tokenizationindex = joblib.load(punkt_tokenizationsindexfile)
    punctword_tokenizationindex = joblib.load(punctword_tokenizationsindexfile)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").drop_duplicates("Id")

    missing = predictions.index[predictions.Tags.isnull()]
    logger.info("{0} Computing {1} missing tags between {2} and {3}...".format(asctime(), len(missing),
                                                                               predictions.Id[missing[0]],
                                                                               predictions.Id[missing[-1]]))
    punkt_tokenizations = pd.Series()
    wordpunct_tokenizations = pd.Series()
    counter = 0
    for i in missing:
        counter += 1
        if counter % 10000 == 0:
            logger.info(asctime() + " Done: {0} out of {1}.".format(counter, len(missing)))
        if predictions.Id[i] not in punkt_tokenizations.index:
            logger.info(asctime() + " Loading tokenizations for {0} from '{1}'".format(predictions.Id[i],
                                                                                       punkt_tokenizationindex[
                                                                                           predictions.Id[i]]))
            punkt_tokenizations = joblib.load(punkt_tokenizationindex[predictions.Id[i]])
            logger.info(asctime() + " Loading tokenizations for {0} from '{1}'".format(predictions.Id[i],
                                                                                       punctword_tokenizationindex[
                                                                                           predictions.Id[i]]))
            wordpunct_tokenizations = joblib.load(punctword_tokenizationindex[predictions.Id[i]])
            logger.info(asctime() + " Done reading '{0}'.".format(punctword_tokenizationindex[predictions.Id[i]]))
        tokenization = pd.Series(Counter(punkt_tokenizations[predictions.Id[i]].to_dict()) + Counter(
            wordpunct_tokenizations[predictions.Id[i]].to_dict()))
        predictions.Tags[i] = " ".join(find_tags(tokenization, tags, multitoken_i))

    outfile = "/home/carsten/facebook/predictions_{0}documents.csv".format(nrows)
    logger.info(asctime() + " Writing predictions to '{0}'...".format(outfile))
    predictions.sort(columns="Id").to_csv(outfile, index=False, cols=["Id", "Tags"], quoting=csv.QUOTE_ALL)
    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"

    columns = ["Id", "Title", "Tags"]
    tagcache = "/home/carsten/facebook/cache/tags"
    #tokenizationfile = "/home/carsten/facebook/cache/tokenizations"
    #tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex"
    nrows = None

    punkt_tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex"
    punctword_tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex_wordpunct"
    if nrows is not None:
        punkt_tokenizationsindexfile += str(nrows)
    punctword_tokenizationsindexfile += str(nrows)

    main()

