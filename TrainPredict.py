import logging
import time
import csv
import sys

from sklearn import multiclass, feature_extraction, preprocessing, tree

from Utils import *


__author__ = 'carsten'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

trainingzipfile = "/home/carsten/facebook/Train.zip"
trainingfile = "Train.csv"
testzipfile = "/home/carsten/facebook/Test.zip"
testfile = "Test.csv"
outfile = "/home/carsten/facebook/predictions.csv"

use_idf = True
smooth_idf = True

lines_training = 100
lines_test = 1000

processes = -1

### Features for Hashing Vectorizer
#n_features = 2**16
n_features = 2 ** 13


def taglist(dataframe):
    tags = set()
    for question in dataframe.Tags:
        tags.update(question)
        for tag in question:
            tags.update(tag.split("-"))
    return list(tags)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            lines_training = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid number of training samples to read: " + sys.argv[1])
            sys.exit()
    outfile += "_" + str(lines_training)

    logger.info(time.asctime() + "\tReading training data (first {0} samples)...".format(lines_training))

    data, texts = read_data(trainingzipfile, trainingfile, lines_training)
    vocabulary = taglist(data)


    ### Tf-IDF full vocabulary
    logger.info(time.asctime() + "\tComputing document vectors (using Tf-Idf)...")
    #vectorizer = feature_extraction.text.TfidfVectorizer(max_df=1.0, min_df=0, use_idf=use_idf,
    #                                                     smooth_idf=smooth_idf, tokenizer=NLTKTokenizer())

    ### Tf-IDF tags as vocabulary
    vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=vocabulary, use_idf=use_idf, smooth_idf=smooth_idf,
                                                         tokenizer=NLTKTokenizer())
    X = vectorizer.fit_transform(texts).toarray()

    ### HashingVectorizer
    #logger.info(time.asctime()+"\tComputing document vectors (using Hash Vectorization)...")
    #vectorizer = feature_extraction.text.HashingVectorizer(tokenizer=NLTKTokenizer(vocabulary), n_features=n_features, norm="l1")
    #X = vectorizer.transform(texts).toarray()

    logger.info(time.asctime() + "\tBinarizing labels...")
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(data.Tags)

    logger.info(time.asctime() + '\tTraining classifier...')
    starttime = time.time()
    #estimator = linear_model.SGDClassifier()
    #estimator = naive_bayes.MultinomialNB(fit_prior=True)
    estimator = tree.DecisionTreeClassifier()

    classifier = multiclass.OneVsRestClassifier(estimator, processes)
    #classifier = multiclass.OneVsOneClassifier(estimator, processes)
    classifier.fit(X, y)

    logger.info(time.asctime() + '\tReading test data...')
    test, test_texts = read_data(testzipfile, testfile, lines_test)

    logger.info(time.asctime() + '\tVectorizing test data...')
    test_X = vectorizer.transform(test_texts).toarray()

    logger.info(time.asctime() + "\tCalculating predictions...")
    predictions = classifier.predict(test_X)
    labels = lb.inverse_transform(predictions)

    logger.info(time.asctime() + '\tWriting predictions for {1} entries to "{0}"'.format(outfile, len(test)))
    with open(outfile, "wb") as csvfile:
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(("Training Samples: " + str(lines_training), "Vocabulary Size: " + str(len(vocabulary))))
        csvwriter.writerow(("Id", "Tags", "Question Title"))
        for i in range(len(test)):
            csvwriter.writerow((test.Id[i], " ".join(labels[i]), test.Title[i]))
