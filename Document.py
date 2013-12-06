from string import punctuation
from nltk import wordpunct_tokenize, sent_tokenize, clean_html, PorterStemmer, corpus
from collections import Counter
from pandas import Series
from multiprocessing import Pool

__author__ = 'carsten'

_stemmer = PorterStemmer()
_token_separator = "#"


def count_tokens(text, tag=None, language="english"):
    tokens = Counter()
    sentences = sent_tokenize(clean_html(text).lower())
    for sentence in sentences:
        for token in wordpunct_tokenize(sentence):
            if token not in corpus.stopwords.words(language) and token not in punctuation:
                stem = _stemmer.stem(token)
                if tag is not None:
                    stem += _token_separator + tag
                tokens[stem] += 1
    return tokens


def vectorize_row(row, language="english"):
    tokens = count_tokens(row[1], tag="t", language=language)
    tokens.update(count_tokens(row[2], tag="b", language=language))
    return Series(tokens)


def vectorize(rows, language="english"):
    pool = Pool()
    vectors = pool.map(vectorize_row, rows) # TODO: language
    pool.close()
    pool.join()
    return vectors
