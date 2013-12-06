import pandas as pd
import csv
import logging
import zipfile
from nltk import PorterStemmer
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from time import asctime
from string import punctuation
from collections import Counter
from os import mkdir
from Memoize import Memoize

from joblib import Memory, Parallel, delayed


__author__ = 'carsten'

nrows = 1000    # globally limit number of rows to read in read_zip

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

cachedir = "/home/carsten/facebook_tags_cache_" + str(nrows) + "docs"
logger.info("Creating directory '{0}'.".format(cachedir))
try:
    mkdir(cachedir)
except OSError:
    logger.debug("'{0}' already exists.".format(cachedir))

memory = Memory(cachedir=cachedir, mmap_mode="r", verbose=0)


def read_zip(zipfilename, filename, cols=None, index_col=None):
    logger.info(asctime() + " Reading {0}/{1}...".format(zipfilename, filename))
    z = zipfile.ZipFile(zipfilename)
    content = pd.read_csv(z.open(filename), index_col=index_col, usecols=cols, quoting=csv.QUOTE_ALL, nrows=nrows)
    z.close()
    logger.info(asctime() + " File {0}/{1} read ({2} entries).".format(zipfilename, filename, len(content)))
    return content


def coocurrences(df):
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
    if df is None:
        df = data
    docs = list()
    for i in df.index:
        if token in df.Tokens[i]:
            docs.append(i)
    return docs


def match_tags(tokens):
    tags = list()
    #logger.debug("Tokens: "+str(tokens))
    #docs = reduce(lambda x, y: x + y, map(find_tags, tokens))
    docs = reduce(lambda x, y: x + y, map(find_tags_memory, tokens))
    #logger.debug("Docs: "+str(docs))
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


def tokenize(title):
    return map(_stemmer.stem, filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
                                     punkttokenizer.tokenize(title.lower())))


def main():
    global data
    data = read_zip(trainingzip, trainingfile, cols=columns, index_col=0).drop_duplicates(cols="Title")
    logger.info(asctime() + " Tokenizing titles...")

    data["Tokens"] = data.Title.map(tokenize)
    #data["Tokens"] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(tokenize)(data.Title[i]) for i in data.index)

    logger.info(asctime() + " Splitting tag strings...")
    data.Tags = data.Tags.map(str.split)
    #data["Tags"] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(delayed(str.split)(data.Tags[i]) for i in data.index)

    test = read_zip(testzip, testfile, cols=columns)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").sort(columns=["Id"]).drop_duplicates("Id")

    logger.info(asctime() + " Tokenizing {0} entries with missing tags...".format(sum(predictions.Tags.isnull())))
    predictions["Tokens"][predictions.Tags.isnull()] = predictions.Title[predictions.Tags.isnull()].map(tokenize)
    #predictions["Tokens"][predictions.Tags.isnull()] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(tokenize)(predictions.Title[i]) for i in predictions.index[predictions.Tags.isnull()])

    logger.info(asctime() + " Computing missing tags using titles...")
    predictions.Tags[predictions.Tags.isnull()] = predictions.Tokens[predictions.Tags.isnull()].map(match_tags)
    #predictions.Tags[predictions.Tags.isnull()] = Parallel(n_jobs=cpus, verbose=parallel_verbosity)(
    #    delayed(match_tags)(predictions.Tokens[i]) for i in predictions.index[predictions.Tags.isnull()])

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

    #outfile = "/home/carsten/facebook/predictions.csv"
    outfile = "/tmp/predictions.csv"


    max_tags = 5    # maximum tags to assign in match_tags
    cpus = 2    # cpus to use when iterating over rows without tags
    tagcounts = None    # Series containing all tagcounts
    columns = ("Id", "Title", "Tags")
    parallel_verbosity = 5
    find_tags_memory = Memoize(find_tags, nrows)

    main()


