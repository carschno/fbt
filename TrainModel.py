import logging
import time
from nltk import PorterStemmer

from sklearn import multiclass, feature_extraction, linear_model
from sklearn.externals import joblib
from TrainPredict import taglist

from Utils import *


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

_stemmer = PorterStemmer()
_token_separator = "#"

zipfile = "/home/carsten/facebook/Train.zip"
trainingfile = "Train.csv"

#trainingfile = "/home/carsten/facebook/Train_130000.csv"
modelfile = "/home/carsten/facebook/model/svc.bin"
vocabfile = "/home/carsten/facebook/model/vocabulary"
binarizerfile = "/home/carsten/facebook/model/binarizer"

lines = 10000
max_df = 1.0
min_df = 0.1
use_idf = True
smooth_idf = True
processes = -1
n_features = 2 ** 16





if __name__ == "__main__":
    logger.info("Reading training data...")

    data, texts = read_data(zipfile, trainingfile, lines)
    vocabulary = taglist(data)

    logger.info("Computing document vectors...")
    #vectorizer = feature_extraction.text.TfidfVectorizer(stop_words="english", max_df=max_df, use_idf=use_idf,
    #                                                     smooth_idf=smooth_idf, tokenizer=NLTKTokenizer())
    #vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=vocabulary, use_idf=use_idf, smooth_idf=smooth_idf)
    #X = vectorizer.fit_transform(texts)

    vectorizer = feature_extraction.text.HashingVectorizer(tokenizer=NLTKTokenizer(vocabulary), n_features=n_features)
    X = vectorizer.transform(texts)

    logger.info('Training classifier...')
    starttime = time.time()
    estimator = linear_model.SGDClassifier()
    #estimator = sklearn.naive_bayes.GaussianNB()

    classifier = multiclass.OneVsRestClassifier(estimator, processes)
    #lb = preprocessing.LabelBinarizer()
    classifier.fit(X, data.Tags)

    logger.info('Classifier trained ({0}s)'.format(time.time() - starttime))

    logger.info('Saving model in "{0}" and vocabulary in "{1}"...'.format(modelfile, vocabfile))
    joblib.dump(classifier, modelfile)
    joblib.dump(vectorizer.vocabulary_, vocabfile)
