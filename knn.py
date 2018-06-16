#!/usr/bin/env python
"""
knn.py is a python 2.7 script implementing binary 2-dimensional k-nearest neighbors
clustering, to spec of Stanford class BMI 214 Project 2, with n-fold cross validation

@author: Will Connors

usage:
python knn.py posData negData k p n
    INPUTS:
        posData :: string; filepath to TSV integer vectors of positive data samples
        negData :: string; filepath to TSV integer vectors of positive data samples
        k :: int; Number of labelled neighbors to consider for classification [0+]
        p :: float; threshold fraction neighbors positive for positive labeling [0-1]
        n :: int; number of validation folds; withhold 1 of every n samples for testing
    OUTPUTS:
        file ./knn.out :: reports performance statistics (acc,sens,spec) and k,p,n inputs
some explicit data assumptions:
- columns correspond to samples (for example patients) and rows correspond
to the values used to represent the samples (for example gene expression values)
-the rows in each of the 2 files are in the same order
-the values have all been normalized
"""

from __future__ import absolute_import, division, print_function
import sys
import numpy as np


def main(pos_data, neg_data, k, p, n):
    """ knn.py is a python 2.7 script implementing binary 2-dimensional
      k-nearest neighbors clustering, to spec of Stanford class BMI 214
      Project 2, with n-fold cross validation.
      INPUTS:
        pos_data :: string; filepath to TSV integer vectors of positive data samples
        neg_data :: string; filepath to TSV integer vectors of positive data samples
        k :: int; [0+]; Number of labelled neighbors to consider for classification
        p :: float; [0-1]; threshold fraction neighbors positive for positive labeling
        n :: int; number of validation folds,ie withhold 1 of every n samples for testing
      OUTPUTS:
        ./knn.out :: file; reports performance statistics (acc,sens,spec) and k,p,n inputs
    """
    vectors, positives, negatives = preprocess(pos_data, neg_data)
    folds = dividenfold(n, positives, negatives)
    labels = [1 for x in positives]
    labels.extend([0 for y in negatives])

    TP, TN, FP, FN = runfolds(labels, folds, vectors, (k, p, n))

    # calculate desired stats
    # Sensitivity = TP/(TP+FN); Specificity = TN/(TN+FP); Accuracy = (TP+TN)/total
    acc = (TP + TN) / (TP + TN + FP + FN)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    stats = [k, p, n, acc, sens, spec]

    dataout(*stats)

    return


def preprocess(pos_file, neg_file):
    """ function preprocess(), called by main(), reads in from data files and
      assigns vector point ids.
      INPUT:
        pos_file :: string; filepath to positive vectors
        neg_file :: string; filepath to negative vectors
      OUTPUT:
        vect_dict :: dict<int--numpy.ndarray>; maps id numbers to vector position
        truth_ids :: list<int>; id numbers of positive labelled vectors
        neg_ids :: list<int>; id numbers of negative labelled vectors
    """
    try:
        pos = np.loadtxt(pos_file, dtype=int, delimiter='\t', unpack=True)
        truth_ids = []
        vect_dict = {}
        for i, vect in enumerate(pos):
            truth_ids.append(i)
            vect_dict[i] = vect
        offset = len(truth_ids)

        neg = np.loadtxt(neg_file, dtype=int, delimiter='\t', unpack=True)
        neg_ids = []
        for j, vect in enumerate(neg):
            neg_ids.append(j+offset)
            vect_dict[j+offset] = vect

        return vect_dict, truth_ids, neg_ids

    except IOError:
        raise IOError('Problem reading in vectors from files!')


def dividenfold(n, pos_names, neg_names):
    """ function dividenfold(), called by main(), prepares n-fold cross validation
      of data by dividing the positive and negative ids into n separate bins.
      INPUTS:
        n :: int; number of data bins
        pos_names :: list<int>; id numbers of positive vectors
        neg_names :: list<int>; id numbers of negative vectors
      OUTPUTS:
        folds_dict :: dict<int -- set<ints>>; dictionary container of n groups of ids
    """

    pos_unordered = np.random.permutation(pos_names)
    neg_unordered = np.random.permutation(neg_names)
    pairs = zip(pos_unordered, neg_unordered)

    # each dict entry holds a disjoint list of ids
    folds_dict = {}
    for i in range(n):
        folds_dict[i] = []
    for j, pair in enumerate(pairs):
        folds_dict[j % n].extend(pair)

    # logic to distribute any unmatched/uneven label numbers; all neg ids after pos ids
    if len(pos_names) > len(neg_names):
        leftoff = len(neg_names)
        for k, singleton in enumerate(pos_unordered[leftoff:], start=len(neg_names)):
            folds_dict[k % n].append(singleton)
    elif len(pos_names) < len(neg_names):
        leftoff = 2 * len(pos_names)
        for k, singleton in enumerate(neg_unordered[leftoff:], start=len(pos_names)):
            folds_dict[k % n].append(singleton)

    # switch lists to sets
    for i in range(n):
        folds_dict[i % n] = set(folds_dict[i % n])

    return folds_dict


def runfolds(truth_table, folded_ids, vector_dict, params):
    """function runfolds(), called by main(), prepares unique numpy array for each
      knn run and aggregates statistics from all runs.
      INPUTS:
        truth_table :: list<boolean>; maps vector id (index) to label (0|1)
        folded_ids :: dict< int -- set<int>>; maps vector ids into each fold
        vector_dict :: dict< int -- numpy.ndarray>>; maps vector ids to vector data
        params: tuple;
        k :: int; number of neighbors to vote on classification
        p :: float; fraction threshold of positive neighbors to label a new positive
        n :: int; number of folds and knn runs
      OUTPUTS:
        agg_stats :: tuple<int>; number of true positives, true negatives,
          false positives, and false negatives over all runs
    """

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    full_ids = set(range(len(truth_table)))

    # n-folds loop
    for i in range(len(folded_ids)):
        test_ids = folded_ids[i]
        train_ids = full_ids - test_ids
        # switch to list to preserve ordering
        test_ids = list(test_ids)
        train_ids = list(train_ids)
        test_truths = [truth_table[x] for x in test_ids]
        train_truths = [truth_table[y] for y in train_ids]

        # build data matrix
        test_pts = [vector_dict[x] for x in test_ids]
        train_pts = np.asarray([vector_dict[y] for y in train_ids], dtype=np.int_)
        training_data = train_truths, train_pts

        test_classified = runknn(training_data, test_pts, params)

        # check classifications
        for truth, label in zip(test_truths, test_classified):
            if truth and label:
                TP += 1
            elif truth:
                FN += 1
            elif label:
                FP += 1
            else:
                TN += 1

    agg_stats = TP, TN, FP, FN
    return agg_stats


def runknn(training, test_list, params):
    """function runknn(), called my runfolds(), implements the heart of the knn
      classification algorithm.
        INPUTS:
          training :: tuple<list<boolean>,np.ndarray>; package of training points
            and order-preserved labels
          test_list :: list<np.ndarray>; the coordinates of unlabeled points to classify
          params :: tuple; contains input arguments:
            k :: int; Number of labelled neighbors to consider for classification [0+]
            p :: float; threshold fraction neighbors positive for positive labeling [0-1]
            n :: int; number of validation folds; withhold 1 of every n samples for testing
        OUTPUTS:
          classifications :: list<boolean>; order-preserved pos/neg labels of
            test_list points
    """

    known_labels, matrix = training
    k, p, n = params
    classifications = []

    for test in test_list:
        # Compute the distance to all labelled data
        distances = np.linalg.norm(matrix - test, axis=1)
        distances = np.stack((distances, known_labels), axis=-1)
        # Sort the distances, keeping paired label
        distances = distances[distances[:, 0].argsort()]
        # Find closest k labelled data
        closest = distances[:k, :]
        # Use p threshold vote of labels to decide on test label
        if np.sum(closest[:, 1]) / k >= p:
            classifications.append(1)
        else:
            classifications.append(0)

    return classifications


def dataout(k, p, n, accuracy, sensitivity, specificity):
    """function dataout(), called by main(), handles formatting and writing to file
      statistics of classifier performance.
      INPUTS:
        k :: int; Number of labelled neighbors to consider for classification [0+]
        p :: float; threshold fraction neighbors positive for positive labeling [0-1]
        n :: int; number of validation folds; withhold 1 of every n samples for testing
        accuracy :: float; classification metric of how often correct
        sensitivity :: float; classification metric of how many positives it misses
        specificity :: float; classification metric of how many false positives
      OUTPUTS:
        ./knn.out :: file; details input arguments such as k,p,n, and records
          algorithm performance metrics, such as accuracy
    """
    try:
        with open('knn.out', 'w') as f:
            f.write("k: " + str(k) + '\n')
            f.write("p: " + "%.2f" % p + '\n')
            f.write("n: " + str(n) + '\n')
            f.write("accuracy: " + "%.2f" % accuracy + '\n')
            f.write("sensitivity: " + "%.2f" % sensitivity + '\n')
            f.write("specificity: " + "%.2f" % specificity)

    except IOError:
        raise IOError('Problem writing results to file! File locked?')


# ------------- SCRIPT execution -------------
if __name__ == '__main__':
    # check right number of args
    if len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]))
        exit(0)

    else:
        raise SyntaxError('Incorrect number of arguments!', sys.argv)
