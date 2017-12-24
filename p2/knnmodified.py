#!/usr/bin/env python
"""
Will Connors

KNN classifier for ALL/AML [binary] genetic expression microarray data with n-fold cross validation
"""
import numpy as np;
import sys;


def main(positives, negatives, k, p=0.5, n):


    '''
    Inputs:
        positives :: tsv file location of positive integer expression readout,
        negatives :: tsv file location of negative integer expression readout,
        k :: integer value of number of training points to compare with in KNN
        p :: 0 - 1 'burden of evidence' fraction, % of positive neighbors threshold for positive assignment
    Outputs:
        console output :: prints k,p,n values and algorithm performance (accuracy, specificity, sensitivity),
        file write out :: file 'knn.out' in working dir mirrors console output,
        returns :: None
    '''
# Initialize, read in data
try:
    ALL = np.loadtxt(positives, dtype=int, delimiter='\t', unpack=True);
    AML = np.loadtxt(negatives, dtype=int, delimiter='\t', unpack=True);
    k = int(k);
    p = float(p);
    n = int(n);
except(Exception):
    print('File read in error! Check files and required arguments');
    raise Exception

TP, FP, TN, FN = 0, 0, 0, 0;

# build n data sets
sets = nFold(n, ALL, AML);
# run all data sets
for i in range(n):
    results = knn(sets[i, 0], sets[i, 1], sets[i, 2]);
    # check results TODO
    for j in range(0, len(results), 2):
        if (results[j + 1] == 'P'):
            for z in pos_dict[pos_order[i]]:
                if (results[j] == z).all():
                    TP += 1;

            for z in neg_dict[neg_order[i]]:
                if (results[j] == z).all():
                    FP += 1;

        elif (results[j + 1] == 'N'):
            for z in pos_dict[pos_order[i]]:
                if (results[j] == z).all():
                    FN += 1;

            for z in neg_dict[neg_order[i]]:
                if (results[j] == z).all():
                    TN += 1;

# report/record results
return fileOut(k, p, n, TP, FP, TN, FN)


def knn(pos_training, neg_training, test):


    '''
    KNN algorithm implementation
    Inputs:
        pos_training :: An array of positive points,
        neg_training :: an array of negative points,
        test :: an array of test points (each point is an np array of ndim)
    Output:
        classification :: A list alternating between a given test point(an array) and its ALL classification ('N' or 'P')

    '''
classification = [];

for point in test:
    dist_map = {};
    vote = 0.;

    # Compute the euclidean distance to all labelled data
    for train_point in pos_training:
        dist_map[np.sqrt(np.sum((train_point - point) ** 2, dtype=np.int64))] = 'P';
    for train_point in neg_training:
        dist_map[np.sqrt(np.sum((train_point - point) ** 2, dtype=np.int64))] = 'N';

    # Sort the distances
    sorted_distances = dist_map.keys().sort();

    # Find closest k labelled data
    for i in range(k):
        if (dist_map[sorted_distances[i]] == 'P'):
            vote += 1;
    vote_frac = vote / k;

    # Use majority vote of K to decide on label
    classification.append(point);
    if (vote_frac > p):
        classification.append('P');
    else:
        classification.append('N');

return classification;


def nFold(n, pos, neg):


    '''
    n-fold cross validation
    Inputs:
        n ::
        pos ::
        neg ::
    Outputs:
        folds :: n-length list of 3-tuple (pos, neg, test) arrays of np ndim data points
    '''

# Randomize the positive data and divide it into n equally (or as close to equal as possible) sized sets.
np.random.shuffle(ALL);
pos_dict = {};
for i in range(n):
    pos_dict[i] = ALL[i::n];

# Repeat for negative data
np.random.shuffle(AML);
neg_dict = {};
for i in range(n):
    neg_dict[i] = AML[i::n];

# Pair up the positive and negative data sets so that you have n pairs that contain approximately equal proportions of positive and negative examples.
# Each pair consists of one positive data set, and one negative data set.
pos_order = list(range(n));
np.random.shuffle(pos_order);
neg_order = list(range(n));
np.random.shuffle(neg_order);

# Take n-1 of the n pairs and combine them to make the training set.
# Run KNN on this training set using the remaining one (out of the four) pair as a test set
# Repeat the previous step three more times. Each time one of the four pairs is left out as the test set, and the remaining 3 pairs are used as the training set.
for i in range(n):
    test_set = [];
    pos_set = [];
    neg_set = [];
    test_set = list(pos_dict[pos_order[i]]) + list(neg_dict[neg_order[i]]);
    pos_training_dict = pos_dict.copy();
    del pos_training_dict[pos_order[i]];
    neg_training_dict = neg_dict.copy();
    del neg_training_dict[neg_order[i]];
    for value in pos_training_dict.values():
        for patient in value:
            pos_set.append(patient);
    for value in neg_training_dict.values():
        for patient in value:
            neg_set.append(patient);

return folds


def fileOut(k, p, n, TP, FP, TN, FN):


    '''
    function to handle performance reporting back to user console and file
    Inputs:
        k ::
        p ::
        n ::
        TP ::
        FP ::
        TN ::
        FN ::
    Outputs:
        console ::
        knn.out ::
        returns None
    '''
##KNN output
print("k: ", k);
print("p: ", "%.2f" % p);
print("n: ", n);
# Sensitivity = TP/(TP+FN)
# Specificity = TN/(TN+FP)
# Accuracy = (TP+TN)/total
acc = (TP + TN) / (TP + TN + FP + FN);
sens = TP / (TP + FN);
spec = TN / (TN + FP);
print("accuracy: ", "%.2f" % acc);
print("sensitivity: ", "%.2f" % sens);
print("specificity: ", "%.2f" % spec);

with open('knn.out', 'w') as f:
    f.write("k: " + str(k) + '\n');
    f.write("p: " + "%.2f" % p + '\n');
    f.write("n: " + str(n) + '\n');
    f.write("accuracy: " + "%.2f" % acc + '\n');
    f.write("sensitivity: " + "%.2f" % sens + '\n');
    f.write("specificity: " + "%.2f" % spec);

return None

# Script execution
if __name__ == '__main__':
    main(*sys.argv[1:]);
