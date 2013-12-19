import pandas as pd
import csv
import logging
from time import asctime
import itertools
import joblib

from Utils import read_zip


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def find_tags(document, tags, mwe_i=dict(), max_tags=5):
    """
    Compute the n most likely tags for the given bag of wordsin token_series.
    @param document: a series representing a document bag of words
    @type document: pd.Series
    @param tags: all available tags and their frequencies
    @type tags: pd.Series
    @param mwe_i: a map from tokens to mwe tags containing them
    @type mwe_i: dict
    @param max_tags: maximum number or tags assigned to text
    @type max_tags: 5
    @return: a list of tags assigned to the document
    @rtype: list
    """

    counts = (pd.Series(document) * tags[document.keys()]).dropna()

    # normalize frequencies:
    #counts = (pd.Series(token_series.astype(np.float) / token_series.sum()) * tags[token_series.keys()]).dropna()

    counts = pd.concat([counts, find_mwe_tags(document, tags, mwe_i)])
    counts.sort(ascending=False)
    counts = counts[counts.notnull()][:max_tags]
    result = list(counts.drop(counts.index[counts < counts.mean()]).index)
    return result


def find_mwe_tags(document, tags, mwe_i):
    # find multi word tags TODO: optimize speed
    mwecounts = dict()
    # iterate over all document tokens that occur in mwe index
    for token in itertools.ifilter(lambda x: x in mwe_i, document.index):
        # iterate over all mwe tags that contain token
        for tag in mwe_i[token]:
            if all(subtag in document.index for subtag in tag.split("-")):
                # FIXME: this is computed multiple times (for each token in mwe tag)
                mwecounts[tag] = tags[tag] * (sum(document[tag.split("-")]) / len(tag.split("-")))
    return pd.Series(mwecounts)


def mwes(tags, char="-"):
    """
    Find all multi-word tags in s' index, using char as a separator. E.g.: visual-studio-2010
    @param tags: a Series of tags
    @type tags: pd.Series
    @param char: character to detect token separator
    @type char: str
    @return: a list of lists, containing the tokens of multi-token tags
    @rtype: pd.Index
    """

    def splitchar(s):
        return s.split(char)

    mwe_list = filter(lambda x: char in x, tags.index)
    return pd.Series(map(splitchar, mwe_list), index=mwe_list)
    #return pd.Series(tags[mwe_index], index=mwe_index)


def mwe_index(tags, char="-"):
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

    logger.info(asctime() + " Reading tag counts from '{0}'...".format(tagcache))
    tags = joblib.load(tagcache, mmap_mode="r")
    # TODO: filter infrequent tags?
    tags = tags[:len(tags) * 3 / 4]   # remove least frequent tags

    #mwe_tags = mwes(tags)
    mwe_i = mwe_index(tags)

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
    counter = 0
    # normalize tag frequencies
    #tags = tags.astype(np.float) / tags.sum()
    for i in missing:
        counter += 1
        if counter % 10000 == 0:
            logger.info(asctime() + " Done: {0} out of {1}.".format(counter, len(missing)))
            #predictions.Tags[i] = " ".join(find_tags(predictions.Title[i] + " " + clean_html(predictions.Body[i])))
        #print tokenizations[predictions.Id[i]]
        if predictions.Id[i] not in tokenizations.index:
            logger.info(asctime() + " Loading indizes for {0} from '{1}'".format(predictions.Id[i],
                                                                                 tokenizationindex[predictions.Id[i]]))
            tokenizations = joblib.load(tokenizationindex[predictions.Id[i]])
            logger.info(asctime() + " Done reading '{0}'.".format(tokenizationindex[predictions.Id[i]]))
            #predictions.Tags[i] = " ".join(find_tags(tokenizations[predictions.Id[i]], tags, mwe_tags))
        predictions.Tags[i] = " ".join(find_tags(tokenizations[predictions.Id[i]], tags, mwe_i))

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

    tokenizationsindexfile = "/home/carsten/facebook/cache/tokenizationsindex"
    if nrows is not None:
        tokenizationsindexfile += str(nrows)

    main()

