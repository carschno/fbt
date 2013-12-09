import pandas as pd
import numpy as np
import csv
import logging
import zipfile
from nltk import PorterStemmer, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from time import asctime
from string import punctuation
from Utils import read_zip

__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def match_tags(sentence):
    """
    Find the tags contained in the documents also containing any of the given tokens and compute the most frequent ones.
    @param sentence: A sentence (document title) to be matched.
    @type sentence: string
    @return: a list of tags
    @rtype list(string)
    """
    tag_series = filter(lambda x: x is not None, map(cache.get, tokenize(sentence)))

    if len(tag_series) == 0:
        logger.debug(asctime() + " Nothing found for '{0}'. Using tokens from wordpunct_tokenize.  ".format(sentence))
        tag_series = filter(lambda x: x is not None, map(cache.get, wordpunct_tokenize(sentence.lower())))
        if len(tag_series) == 0:
            logger.debug(asctime() + " Failed.")
            return list()

    tags = pd.concat(tag_series).groupby(level=0).sum()

    # Find most likely tags in tag series
    tags.sort(ascending=False)
    tags = tags[:max_tags]
    tags = list(tags.drop(tags.index[tags < tags.mean()]).index)
    return tags


def build_cache(s, sample_portion=0.05, min_sample=1000, n_status=10):
    """
    Construct an inverted index tokens/stems -> document indexes
    @param s: a series of texts
    @type s: pd.Series
    @param n_status: Number of intermediate status reports during index building
    @type n_status: int
    @return:    a dictionary mapping from tokens/stems for document lists
    @rtype  dict
    """
    index = dict()
    limit = int(len(s) * sample_portion) if len(s) * sample_portion > min_sample else min_sample
    logger.info(
        asctime() + " Building inverted index from {0} documents, sample size: {1}".format(len(s), limit))
    counter = 0

    for i in np.random.permutation(s.index)[:limit]:
        counter += 1
        if counter % int(limit / n_status) == 0:
            logger.info(asctime() + " {0} of {1} sample documents processed.".format(counter, limit))
        for token in tokenize(s[i]):
            tags = pd.Series(1, data.Tags[i])
            if token in index:
                index[token] = pd.concat([index[token], tags])  #.groupby(level=0).sum()
            else:
                index[token] = tags

                # if token not in index:
                #     index[token] = Counter()
                # index[token].update(data.Tags[i])
    logger.info(asctime() + " Inverted index contains {0} tokens (stems).".format(len(index)))
    return index    # TODO make this map to a Counter(tags)


def tokenize(sentence, stem=True):
    """
    Tokenize given text, assuming it is a sentence already. Remove stopwords and punctuation and reduce to stems.
    @param sentence:
    @type sentence: str
    @return: a list of tokens
    @rtype: list(str)
    """
    # TODO: instead of stemming, try tokenizing and filtering only. Thus, spare the tokenization of all titles and
    # only check using data.Title.str.contain
    #    return map(_stemmer.stem, filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
    #                                     punkttokenizer.tokenize(title.lower())))
    #tokens = filter(lambda x: x not in stopwords.words("english") and x not in punctuation, punkttokenizer.tokenize(title.lower()))
    tokens = filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
                    word_tokenize(sentence.lower()))
    return map(stemmer.stem, tokens) if stem else tokens


def main():
    """
    Read training data and test data, create inverted index, merge data and test data on duplicate titles, compute
     missing tags using the inverse index, write output to file
    """
    global data
    global cache
    data = read_zip(trainingzip, trainingfile, cols=columns, index_col=0, count=nrows).drop_duplicates(cols="Title")

    logger.info(asctime() + " Splitting tag strings...")
    data.Tags = data.Tags.map(str.split)

    cache = build_cache(data.Title)

    test = read_zip(testzip, testfile, cols=columns)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left", count=nrows).sort(columns=["Id"]).drop_duplicates("Id")

    missing = predictions.index[predictions.Tags.isnull()]
    logger.info(asctime() + " Computing {0} missing tags using titles...".format(len(missing)))

    # FIXME: job.Parallel seems to get stuck here. Due to global variable data?
    #predictions.Tags[missing] = Parallel(n_jobs=cpus, verbose=5)(delayed(match_tags)(predictions.Title[i]) for i in missing)

    #predictions.Tags[missing] = predictions.Title[missing].map(match_tags)
    #pool = Pool(cpus)
    #predictions.Tags[missing] = pool.map(match_tags, predictions.Title[missing])
    #pool.close()
    #pool.join()

    counter = 0
    for i in missing:
        counter += 1
        if counter % 1000 == 0:
            logger.info(asctime() + " {0} of {1} documents processed.".format(counter, len(missing)))
        predictions.Tags[i] = match_tags(predictions.Title[i])

    # TODO consider bodies (on top of titles)

    logger.info(asctime() + " Joining tag lists...")
    predictions.Tags = predictions.Tags.map(" ".join)

    outfile = "/home/carsten/facebook/predictions_{0}documents_{1}tokens.csv".format(nrows, len(cache))

    logger.info(asctime() + " Writing predictions to '{0}'...".format(outfile))
    predictions.to_csv(outfile, index=False, cols=("Id", "Tags"), quoting=csv.QUOTE_ALL)
    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    stemmer = PorterStemmer()

    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"

    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"

    max_tags = 5    # maximum tags to assign in match_tags
    cpus = 2    # cpus to use when iterating over rows without tags
    #    tagcounts = None    # Series containing all tagcounts
    columns = ("Id", "Title", "Tags")
    #    parallel_verbosity = 5
    #    find_tags_memory = Memoize(find_tags, nrows)
    cache = dict()
    nrows = None    # global maximum number of rows to read in read_zip

    main()
