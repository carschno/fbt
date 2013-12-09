import logging
from zipfile import ZipFile
from pandas import read_csv
from nltk import clean_html, PorterStemmer, sent_tokenize, word_tokenize
from time import asctime
import zipfile
import pandas as pd
import csv

__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def read_data(zipfile, csvfile, lines=None):
    z = ZipFile(zipfile)
    data = read_csv(z.open(csvfile), nrows=lines)
    data.Body = data.Body.map(lambda text: text.replace('\n', ' '))
    if "Tags" in data.columns:
        data.Tags = data.Tags.map(str.split)
    texts = (data.Title + " " + data.Body).map(clean_html)
    return data, texts


def read_zip(zipfilename, filename, cols=None, index_col=None, count=None):
    """
    Read a CSV file within a zip archive.
    @param zipfilename: Zip file containing CSV data file
    @param filename: CSV file name within zip file
    @param cols: Columns to read from CSV
    @param index_col: column index to use as index
    @return: a dataframe containing the CSV file's content
    @rtype: pd.DataFrame
    """

    logger.info(asctime() + " Reading up to {2} lines from {0}/{1}...".format(zipfilename, filename, count))
    z = zipfile.ZipFile(zipfilename)
    content = pd.read_csv(z.open(filename), index_col=index_col, usecols=cols, quoting=csv.QUOTE_ALL, nrows=count)
    z.close()
    logger.info(asctime() + " File {0}/{1} read ({2} entries).".format(zipfilename, filename, len(content)))
    return content


class NLTKTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, text):
        tokens = list()
        for sentence in sent_tokenize(text):
            for token in word_tokenize(sentence):
                tokens.append(token.lower())
        return tokens