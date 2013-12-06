import logging

from sklearn.externals import joblib

from sklearn import feature_extraction

from Utils import *


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

zipfile = "/home/carsten/facebook/Test.zip"
testfile = "Test.csv"

modelfile = "/home/carsten/facebook/model/svc.bin"
vocabfile = "/home/carsten/facebook/model/vocabulary"

#binarizerfile = "/home/carsten/facebook/model/binarizer"
lines = 10
use_idf = True
smooth_idf = True
n_features = 2**15

if __name__ == "__main__":
    data, texts = read_data(zipfile, testfile, lines)

    logger.info('Reading model from "{0}" and vocabulary from "{1}"...'.format(modelfile, vocabfile))
    classifier = joblib.load(modelfile)
    vocabulary = joblib.load(vocabfile)

    logger.info('Vectorizing test data....')
    #vectorizer = feature_extraction.text.TfidfVectorizer(stop_words="english", vocabulary=vocabulary, use_idf=use_idf,
    #                                                     smooth_idf=smooth_idf, tokenizer=NLTKTokenizer())
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=vocabulary)
    X = vectorizer.fit_transform(texts).toarray()
    #vectorizer = feature_extraction.text.HashingVectorizer(tokenizer=NLTKTokenizer(), n_features=n_features)
    #X = vectorizer.transform(texts)

    print('Id\tTags')
    for i in range(len(data)):
        print('{0}\t{1}'.format(data.Id[i], classifier.predict(X[i])))


