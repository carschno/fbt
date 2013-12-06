from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import sent_tokenize, wordpunct_tokenize, clean_html
from collections import Counter
from string import punctuation

__author__ = 'carsten'

_token_separator = "#"
stemmer = PorterStemmer()  # TODO: make stemmer language-specific


def question2counter(question, language="english"):
    title, body = question[1:3]
    counter = _text2counter(title, language=language, tag="title")
    counter.update(_text2counter(body, language=language, tag="body"))
    return counter


def _text2counter(text, language="english", tag=None):
    counter = Counter()
    sentences = sent_tokenize(clean_html(text).lower())
    for sentence in sentences:
        for token in wordpunct_tokenize(sentence):
            if token not in stopwords.words(language) and token not in punctuation:
                stem = stemmer.stem(token)
                if tag is not None:
                    stem += _token_separator + tag
                counter[stem] += 1
    return counter


def question2tags(question):
    if len(question) > 3:
        return question[3].split(" ")
    else:
        return None


def termlist(doc_counters):
    if len(doc_counters) > 2:
        return termlist(doc_counters[:len(doc_counters) / 2]).union(termlist(doc_counters[len(doc_counters) / 2:]))
    elif len(doc_counters) == 2:
        return set(doc_counters[0].keys()).union(doc_counters[1].keys())
    elif len(doc_counters) == 1:
        return set(doc_counters[0].keys())
    else:
        return set()


def taglist(tag_lists):
    if len(tag_lists) > 2:
        return taglist(tag_lists[:len(tag_lists) / 2]).union(taglist(tag_lists[len(tag_lists) / 2:]))
    elif len(tag_lists) == 2:
        return set(tag_lists[0]).union(tag_lists[1])
    elif len(tag_lists) == 1:
        return set(tag_lists[0])
    else:
        return set()