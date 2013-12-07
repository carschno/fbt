import pandas as pd
import numpy as np
import csv
import logging
import zipfile
from nltk import PorterStemmer
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from time import asctime
from string import punctuation
from collections import Counter
#from os import mkdir
#from Memoize import Memoize
from re import escape

from joblib import Memory


__author__ = 'carsten'

nrows = 10000    # globally limit number of rows to read in read_zip
max_cache = 5000    # cache n first types only, ignore all others

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

### FOR joblib.Memory
#cachedir = "/home/carsten/facebook_tags_cache_" + str(nrows) + "docs"
#logger.info("Creating directory '{0}'.".format(cachedir))
#try:
#    mkdir(cachedir)
#except OSError:
#    logger.debug("'{0}' already exists.".format(cachedir))
#memory = Memory(cachedir=cachedir, mmap_mode="r", verbose=0)
###

def read_zip(zipfilename, filename, cols=None, index_col=None):
    logger.info(asctime() + " Reading {0}/{1}...".format(zipfilename, filename))
    z = zipfile.ZipFile(zipfilename)
    content = pd.read_csv(z.open(filename), index_col=index_col, usecols=cols, quoting=csv.QUOTE_ALL, nrows=nrows)
    z.close()
    logger.info(asctime() + " File {0}/{1} read ({2} entries).".format(zipfilename, filename, len(content)))
    return content


def coocurrences(df):
    """
    Create a full coocurrence matrix.
    @deprecated
    @param df:
    @return:
    """
    logger.info(asctime() + " Generating token list...")
    all_tokens = pd.Series(df.Tokens.sum()).drop_duplicates()
    logger.info(asctime() + " {0} token types found.".format(len(all_tokens)))

    logger.info(asctime() + " Generating tag list...")
    tags = pd.Series(df.Tags.map(str.split).sum()).drop_duplicates()
    logger.info(asctime() + " {0} tags found.".format(len(tags)))

    logger.info(asctime() + " Generating co-occurrence matrix (tokens/tags)...")
    cooc_matrix = pd.DataFrame(index=all_tokens, columns=tags).fillna(0)
    for i in df.index:
        for token in set(df.Tokens[i]): # segfault here if df.Tokens[i] is list?
            for tag in df.Tags[i].split():
                cooc_matrix.ix[token][tag] += 1
    return cooc_matrix[cooc_matrix > 1] # ignore entries <= 1


#@memory.cache
def find_tags(token, df=None):
    """
    Find all documents containing the token in a Tokens column
    @param token:
    @param df:
    @return:
    """
    # TODO: fill cache in advance with random sample?
    global cache
    global cachelimit_logged
    if df is None:
        df = data

    docs = list()
    if token in cache:
        docs = cache[token]
    # elif len(cache) < max_cache:
    #     docs = list(df.index[df.Title.str.contains(escape(token), case=False)])
    #     # for i in df.index:
    #     #     if token in df.Tokens[i]:
    #     #         docs.append(i)
    #     cache[token] = docs
    # elif not cachelimit_logged:
    #     logger.info(asctime() + " Cache limit ({0}) reached.".format(max_cache))
    #     cachelimit_logged = True
    return docs


def match_tags(title):
    """
    Find the tags contained in the documents also containing any of the given tokens and compute the most frequent ones.
    @param title: A sentence (document title) to be matched.
    @type title: string
    @return: a list of tags
    @rtype list(string)
    """
    tags = list()
    docs = reduce(lambda x, y: x + y, map(find_tags, tokenize(title)))
    #docs = reduce(lambda x, y: x + y, map(find_tags_memory, tokens))   # using Memoize
    if len(docs) > 0:
        tags = pd.Series(Counter(data.Tags.reindex(docs).sum()))
        #logger.debug("All tags: "+str(tags))

        #tags = pd.Series(Counter(reduce(lambda x, y: x + y, map(find_tags, tokens))))
        if len(tags) > 0:
            tags.sort(ascending=False)
            tags = tags[:max_tags]
            #logger.debug("Filtered tags: "+str(tags))
            tags = list(tags.drop(tags.index[tags < tags.mean()]).index)
    return tags

def build_cache(s, cache_size=100000):
    """

    @param s: a series of texts
    @rtype s: pd.Series
    @return:
    """
    global cachelimit_logged
    logger.info(asctime() + " Building inverted index from {0} documents with at most {1} entries...".format(len(s), cache_size))
    index = dict()

    for i in np.random.permutation(s.index):   # TODO randomize order
        for token in tokenize(s[i]):
            if token in index.keys():
                index[token].append(i)
            elif len(index) < cache_size:
                index[token] = [i]
            elif not cachelimit_logged:
                logger.info(asctime() +" Inverted index maximum size {0} reached.".format(len(index)))
                cachelimit_logged = True
    return index



def tokenize(title):
    """
    Tokenize given text, assuming it is a sentence already. Remove stopwords and punctuation and reduce to stems.
    @param title:
    @return:
    """
    # TODO: instead of stemming, try tokenizing and filtering only. Thus, spare the tokenization of all titles and
    # only check using data.Title.str.contain
    #    return map(_stemmer.stem, filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
    #                                     punkttokenizer.tokenize(title.lower())))
    return filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
                  punkttokenizer.tokenize(title.lower()))


def main():
    global data
    global cache
    data = read_zip(trainingzip, trainingfile, cols=columns, index_col=0).drop_duplicates(cols="Title")

    #logger.info(asctime() + " Tokenizing titles...")
    #data["Tokens"] = data.Title.map(tokenize)
    #data["Tokens"] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(tokenize)(data.Title[i]) for i in data.index)

    logger.info(asctime() + " Splitting tag strings...")
    data.Tags = data.Tags.map(str.split)
    #data["Tags"] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(delayed(str.split)(data.Tags[i]) for i in data.index)

    cache = build_cache(data.Title, max_cache)

    test = read_zip(testzip, testfile, cols=columns)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").sort(columns=["Id"]).drop_duplicates("Id")

    #logger.info(asctime() + " Tokenizing {0} entries with missing tags...".format(sum(predictions.Tags.isnull())))
    #predictions["Tokens"][predictions.Tags.isnull()] = predictions.Title[predictions.Tags.isnull()].map(tokenize)
    #predictions["Tokens"][predictions.Tags.isnull()] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(tokenize)(predictions.Title[i]) for i in predictions.index[predictions.Tags.isnull()])

    logger.info(asctime() + " Computing missing tags using titles...")
    #predictions.Tags[predictions.Tags.isnull()] = predictions.Tokens[predictions.Tags.isnull()].map(match_tags)
    #predictions.Tags[predictions.Tags.isnull()] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(match_tags)(predictions.Title[i]) for i in predictions.index[predictions.Tags.isnull()])
    predictions.Tags[predictions.Tags.isnull()] = predictions.Title[predictions.Tags.isnull()].map(match_tags)
    # TODO consider bodies + titles

    logger.info(asctime() + " Joining tag lists...")
    predictions.Tags = predictions.Tags.map(" ".join)

    logger.info(asctime() + " Writing predictions to '{0}'...".format(outfile))
    predictions.to_csv(outfile, index=False, cols=("Id", "Tags"), quoting=csv.QUOTE_ALL)
    logger.info(asctime() + " Done.")


if __name__ == "__main__":
    _stemmer = PorterStemmer()
    punkttokenizer = PunktWordTokenizer()

    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    trainingrows = None

    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"
    testrows = None

    outfile = "/home/carsten/predictions_{0}documents_{1}cache.csv".format(nrows, max_cache)

    max_tags = 5    # maximum tags to assign in match_tags
    cpus = 2    # cpus to use when iterating over rows without tags
    #    tagcounts = None    # Series containing all tagcounts
    columns = ("Id", "Title", "Tags")
    #    parallel_verbosity = 5
    #    find_tags_memory = Memoize(find_tags, nrows)
    cache = dict()
    cachelimit_logged = False   # true after cache limit has been reached the first time

    main()
