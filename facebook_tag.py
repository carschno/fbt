import pandas as pd
import numpy as np
import csv
import logging
import zipfile
from nltk import PorterStemmer, word_tokenize
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from time import asctime
from string import punctuation
from collections import Counter
from multiprocessing import Pool

__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def read_zip(zipfilename, filename, cols=None, index_col=None):
    """
    Read a CSV file within a zip archive.
    @param zipfilename: Zip file containing CSV data file
    @param filename: CSV file name within zip file
    @param cols: Columns to read from CSV
    @param index_col: column index to use as index
    @return: a dataframe containing the CSV file's content
    @rtype: pd.DataFrame
    """
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
        for token in set(df.Tokens[i]):     # segfault here if df.Tokens[i] is list?
            for tag in df.Tags[i].split():
                cooc_matrix.ix[token][tag] += 1
    return cooc_matrix[cooc_matrix > 1]     # ignore entries <= 1


def match_tags(sentence):
    """
    Find the tags contained in the documents also containing any of the given tokens and compute the most frequent ones.
    @param sentence: A sentence (document title) to be matched.
    @type sentence: string
    @return: a list of tags
    @rtype list(string)
    """
    tags = list()
    docs = list()
    for token in tokenize(sentence):
        if token in cache:
            docs.extend(cache[token])

    #docs = reduce(lambda x, y: x + y, map(find_tags, tokenize(sentence)))
    #docs = reduce(lambda x, y: x + y, map(find_tags_memory, tokens))   # using Memoize
    if len(docs) > 0:
        #tags = pd.Series(Counter(data.Tags.reindex(docs).sum()))
        tags = pd.Series(Counter(data.Tags[docs].sum()))
        #logger.debug("All tags: "+str(tags))

        #tags = pd.Series(Counter(reduce(lambda x, y: x + y, map(find_tags, tokens))))
        if len(tags) > 0:
            tags.sort(ascending=False)
            tags = tags[:max_tags]
            #logger.debug("Filtered tags: "+str(tags))
            tags = list(tags.drop(tags.index[tags < tags.mean()]).index)
    return tags


def build_cache(s, sample_portion=0.01, min_sample=1000, n_status=10):
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
            if token in index:
                index[token].append(i)
            else:
                index[token] = [i]
    logger.info(asctime() + " Inverted index contains {0} tokens (stems).".format(len(index)))
    return index    # return pd.Series instead of dict?


def tokenize(title, stem=True):
    """
    Tokenize given text, assuming it is a sentence already. Remove stopwords and punctuation and reduce to stems.
    @param title:
    @type title: str
    @return: a list of tokens
    @rtype: list(str)
    """
    # TODO: instead of stemming, try tokenizing and filtering only. Thus, spare the tokenization of all titles and
    # only check using data.Title.str.contain
    #    return map(_stemmer.stem, filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
    #                                     punkttokenizer.tokenize(title.lower())))
    #tokens = filter(lambda x: x not in stopwords.words("english") and x not in punctuation, punkttokenizer.tokenize(title.lower()))
    tokens = filter(lambda x: x not in stopwords.words("english") and x not in punctuation,
                    word_tokenize(title.lower()))
    return map(stemmer.stem, tokens) if stem else tokens


def main():
    """
    Read training data and test data, create inverted index, merge data and test data on duplicate titles, compute
     missing tags using the inverse index, write output to file
    """
    global data
    global cache
    data = read_zip(trainingzip, trainingfile, cols=columns, index_col=0).drop_duplicates(cols="Title")

    logger.info(asctime() + " Splitting tag strings...")
    data.Tags = data.Tags.map(str.split)

    cache = build_cache(data.Title)

    test = read_zip(testzip, testfile, cols=columns)

    logger.info(asctime() + " Merging training data and test data...")
    predictions = pd.merge(test, data, on="Title", how="left").sort(columns=["Id"]).drop_duplicates("Id")

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
    punkttokenizer = PunktWordTokenizer()

    trainingzip = "/home/carsten/facebook/Train.zip"
    trainingfile = "Train.csv"
    trainingrows = None

    testzip = "/home/carsten/facebook/Test.zip"
    testfile = "Test.csv"
    testrows = None

    max_tags = 5    # maximum tags to assign in match_tags
    cpus = 2    # cpus to use when iterating over rows without tags
    #    tagcounts = None    # Series containing all tagcounts
    columns = ("Id", "Title", "Tags")
    #    parallel_verbosity = 5
    #    find_tags_memory = Memoize(find_tags, nrows)
    cache = dict()
    nrows = None    # global maximum number of rows to read in read_zip

    main()
