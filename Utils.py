
from zipfile import ZipFile
from pandas import read_csv
from nltk import clean_html, PorterStemmer, sent_tokenize, word_tokenize
import nltk.corpus

__author__ = 'carsten'


def read_data(zipfile, csvfile, lines=None):
    z = ZipFile(zipfile)
    data = read_csv(z.open(csvfile), nrows=lines)
    data.Body = data.Body.map(lambda text: text.replace('\n', ' '))
    if "Tags" in data.columns:
        data.Tags = data.Tags.map(str.split)
    texts = (data.Title + " " + data.Body).map(clean_html)
    return data, texts


class NLTKTokenizer(object):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, text):
        tokens = list()
        for sentence in sent_tokenize(text):
            for token in word_tokenize(sentence):
                tokens.append(token.lower())
        return tokens