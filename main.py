"""
Zachary McCullough
mccul157@msu.edu
2020-01-13
main file to run
"""

#########
# imports
#########

import sklearn.datasets
load_data = sklearn.datasets.fetch_20newsgroups
from typing import Any, Union, Tuple, List, Dict
import numpy as np
import numba
import string
import math
from Perceptron.Perceptron import *
from copy import deepcopy
from collections import OrderedDict


###########
# functions
###########

def get_data() -> Tuple[Any, Any]:
    # r = ('headers', 'footers', 'quotes')
    return load_data(subset='train'), load_data(subset='test') # , remove=r)


def word_freq(inp: str, result: Union[Dict, None]=None) -> Dict[str, int]:
    """
    Given a string of words, return dictionary of frequencies
    :param inp: any input string
    :param result: None or an existing dictionary
    :return: Dict{word: count}
    """

    # word processing into list of words
    inp = inp.split()
    inp = [x.strip(string.punctuation).lower() for x in inp]

    # default case
    if result is None:
        result = {x: 0 for x in inp}

    # populate for words in string that aren't already there
    for x in inp:
        if x not in result:
            result[x] = 0

    # count
    for word in inp:
        result[word] += 1
    return result


def all_word_freq(inp: List[str], result: Union[Dict, None]=None) -> Dict[str, int]:
    """
    Calculates word frequency like in word_freq but given list of strings first
    :param inp: List[str]
    :param result: None or Dict
    :return: Dict[word: count]
    """

    # base case
    if result is None:
        result = {}

    for s in inp:
        result = word_freq(s, result)

    return result


def vector_form(data: Dict[str, int]) -> np.array:
    """
    Given dictionary convert it into [0, 1, 2, ..., 0] of uniform length
    for dataset to get idea
    :param data: Dict[word: count]
    :return: ([0, 1, 2, ...] where values are sorted, keys)
    """
    keys = list(data.keys())
    keys.sort()
    return np.array([data[x] for x in keys]), keys


def tf_idf(word_freq: List[np.array], classes: List[str]) -> np.array:
    """
    term frequency-inverse document frequency
    Given a vectorized raw word count, apply tf-idf to the weights
    :param word_freq: [[0, 1, ...], [1, 2, ...], ...] each list refers to word
        frequency count of a document
    :param classes: ["doc1", "doc2", .... ] for decoding word frequency arrays
    :return: [[.012, 0.1232, ... ], ... ] tf-idf vectors
    """

    N = float(len(classes))

    result = []
    for doc_wf in word_freq:
        row = []
        for pos, wf in enumerate(doc_wf):
            non_zeros = sum(1 for x in word_freq if x[pos] != 0)  # sum number of documents term is non-zero
            row.append(wf*math.log(N/(1+non_zeros)))
        result.append(row)
    return result


def get_universal_word_bag(train, test):
    """
    Given train and test data gets word frequency of the entire dataset
    :param train: any list of strings
    :param test: any list of strings
    :return: a list of unique values
    """
    return list(set(train) | set(test))


def tf_idf_other(dataset: List[List[str]], labels: List[int]):
    each_word_doc_total = {}
    each_word_categories = {}

    # calculate word frequency in total
    word_freq_foreach_doc = {}
    for k, doc in enumerate(labels):
        if doc not in word_freq_foreach_doc:
            word_freq_foreach_doc[doc] = {}
        # go through each word in kth datapoint
        for word in dataset[k]:
            if word not in word_freq_foreach_doc[doc]:
                word_freq_foreach_doc[doc][word] = 1
            else:
                word_freq_foreach_doc[doc][word] += 1



    # go through each datapoint
    for k, list_words in enumerate(dataset):
        # iterate through words
        for word in list_words:
            # if word not in, set count to 1, and initiate list keeping track
            # of what docs term is in
            if word not in each_word_doc_total:
                each_word_doc_total[word] = 1
                each_word_categories[word] = [labels[k]]
            # word is in, so then add 1 if new doc, and add new doc label
            else:
                label = labels[k]
                if label not in each_word_categories[word]:
                    each_word_doc_total[word] += 1
                    each_word_categories[word].append(label)

    # delete duplicates, get number of different numbers to get # categories
    N = float(len(set(labels)))
    result_foreach_doc_and_words = {}
    # get inverse word freq for all words
    for k, label in enumerate(labels):
        if label not in result_foreach_doc_and_words:
            result_foreach_doc_and_words[label] = {}
        for word in dataset[k]:
            if word not in result_foreach_doc_and_words[label]:
                result_foreach_doc_and_words[label][word] = \
                    word_freq_foreach_doc[label][word]*math.log(N/(each_word_doc_total[word]+1))

    return result_foreach_doc_and_words


def get_top_n_words(tf_idf, n: int):
    result = {}

    flatten = {}
    for d in tf_idf:
        for k in tf_idf[d]:
            if k not in flatten:
                flatten[k] = tf_idf[d][k]
            else:
                if flatten[k] < tf_idf[d][k]:
                    flatten[k] = tf_idf[d][k]

    def __get_max(d):
        keys = list(d.keys())
        max_k = keys[0]
        max_v = d[max_k]
        for key in keys[1:]:
            if d[key] > max_v:
                max_v = d[key]
                max_k = key
        return max_k

    for i in range(n):
        max_k = __get_max(flatten)
        result[max_k] = flatten[max_k]
        del flatten[max_k]
        if len(flatten) == 0:
            break

    return result


def convert_data(train: List[List[str]], test: List[List[str]], labels: List[int], top=10000):
    combined = train+test
    res = tf_idf_other(combined, labels)
    reduced = get_top_n_words(res, top)
    # empty dict of the keys
    words = list(reduced.keys())

    def __convert(data, reduced):
        new_data = []
        for point in data:
            new_p = np.zeros(len(reduced))
            for word in point:
                if word in reduced:
                    new_p[reduced.index(word)] += 1
            new_data.append(new_p)
        return new_data

    return __convert(train, words), __convert(test, words)


def split_words(data):
    temp = [x.split() for x in data]
    result = []
    for row in temp:
        new_row = []
        for word in row:
            new_row.append(word.translate(str.maketrans('', '', string.punctuation)).lower().strip())
        result.append(new_row)
    return result

def main():
    train, test = get_data()
    train_d, test_d = train.data, test.data
    train_labels, test_labels = train.target, test.target
    print('Splitting words')
    train_d, test_d = split_words(train_d), split_words(test_d)
    print('Converting data')
    train_data, test_data = convert_data(train_d, test_d, np.concatenate((train_labels, test_labels), axis=0))

    # model
    print('Training model')
    model = Perceptron(len(train_data[0]))
    model.train(train_data, train_labels, 0.001)
    print('Model Eval: ', end='')
    result = []
    for point in test_data:
        result.append(model.predict(point)[0])
    dif = result-test_labels
    print('Accuracy: ', len(np.where(dif == 0))/len(dif))
    pass


if __name__ == '__main__':
    main()

