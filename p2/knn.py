"""
Will Connors

KNN classifier
"""
import numpy as np;
import sys;


# Performs KNN algorithm
# Inputs: An array of positive points, an array of negative points, and an array of test points (each point is an array of ndim)
# Output:A list alternating between a given test point(an array) and it's classification ('N' or 'P')
def knn(pos_training, neg_training, test):
    classification = [];
    for point in test:
        dist_map = {};
        percent_pos = 0.;
        #	1. Compute the distance to all labelled data
        for train_point in pos_training:
            dist_map[np.sqrt(np.sum((train_point - point) ** 2, dtype=np.int64))] = 'P';
        for train_point in neg_training:
            dist_map[np.sqrt(np.sum((train_point - point) ** 2, dtype=np.int64))] = 'N';
        #	2. Sort the distances
        sorted_distances = np.sort(list(dist_map.keys()));

        #	3. Find closest k labelled data
        for i in range(k):
            if (dist_map[sorted_distances[i]] == 'P'):
                percent_pos += 1;
        percent_pos = percent_pos / k;
        #	4. Use majority vote of K to decide on label
        classification.append(point);
        if (percent_pos > p):
            classification.append('P');
        else:
            classification.append('N');
    return classification;


# python knn.py pos neg k p n

####MAIN PROGRAM#######


# KNN input

ALL = np.loadtxt(sys.argv[1], dtype=int, delimiter='\t', unpack=True);
AML = np.loadtxt(sys.argv[2], dtype=int, delimiter='\t', unpack=True);
k = int(sys.argv[3]);
p = float(sys.argv[4]);
n = int(sys.argv[5]);

# n-fold cross validation
# Randomize the positive data and divide it into 4 equally (or as close to equal as possible) sized sets.

np.random.shuffle(ALL);
pos_dict = {};
for i in range(n):
    pos_dict[i] = ALL[i::n];

# Randomize the negative data and divide it into 4 equally (or as close to equal as possible) sized sets.
np.random.shuffle(AML);
neg_dict = {};
for i in range(n):
    neg_dict[i] = AML[i::n];

# Pair up the positive and negative data sets so that you have 4 pairs that contain approximately equal proportions of positive and negative examples. Each pair consists of one positive data set, and one negative data set.
pos_order = list(range(n));
np.random.shuffle(pos_order);
neg_order = list(range(n));
np.random.shuffle(neg_order);

# Now take three of the four pairs and combine them to make the training set. Run KNN on this training set using the remaining one (out of the four) pair as a test set (i.e. pretend you don't know the classification of the items in the test set, let your algorithm predict the class for each item, then see if the prediction was correct).
# Repeat the previous step three more times. Each time one of the four pairs is left out as the test set, and the remaining 3 pairs are used as the training set.
TP, FP, TN, FN = 0, 0, 0, 0;

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
    results = knn(pos_set, neg_set, test_set);

    # check results
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
