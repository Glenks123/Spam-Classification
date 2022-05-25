import re
import os
import numpy as np
from nltk.stem import PorterStemmer
import sys

sys.path.append('..')


# def guassianKernal(x1, x2, sigma):
#     sim = np.exp((-1 * sum(abs(x1 - x2) ** 2)) / (2 * sigma ** 2))
#     return sim

def emailFeatures(word_indices):
    n = 1899
    x = np.zeros(n)

    for i in range(len(word_indices)):
        x[word_indices[i]] = 1

    return x


def getVocabList():
    vocabList = np.genfromtxt(os.path.join('Data', 'vocab.txt'), dtype=object)
    return list(vocabList[:, 1].astype(str))


def processEmail(email_contents):
    # Load Vocabulary
    vocabList = getVocabList()

    word_indices = []

    email_contents = email_contents.lower()

    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    email_contents = re.compile(
        '(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    email_contents = re.compile(
        '[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    email_contents = re.split(
        '[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    email_contents = [word for word in email_contents if len(word) > 0]
    stemmer = PorterStemmer()
    processed_email = []
    for word in email_contents:
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)
        processed_email.append(word)

        if len(word) < 1:
            continue

        if word in vocabList:
            word_indices.append(vocabList.index(word))

    return word_indices
