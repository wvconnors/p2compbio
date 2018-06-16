#!/usr/bin/env python
"""  kmeans.py is a python 2.7 script implementing kmeans clustering, per the
spec of Stanford class BMI 214, project 2

@author: Will Connors
usage:
  python kmeans.py k expression.dat max.it centroids.txt

"""

from __future__ import absolute_import, division, print_function
import sys
import numpy as np


def main(k, data_file, max_it, centroids_file = None):
    """
    """
    matrix, centroid_loc = configure(k, data_file, centroids_file)
    # build some mappings
    pt_affil = np.empty()
    centroid_posse = np.empty()
    converged_after = max_it
    for w in range(max_it):
        pt_affil, centroid_posse, settled = updateclusters(matrix, centroid_loc,
                                                           pt_affil, centroid_posse)
        centroid_loc = updatecentroids(matrix, centroid_loc, centroid_posse)
        if settled:
            converged_after = w
            break
    # else:
    #     print('Warning, max iterations reached without stable assignments!')

    report(converged_after, pt_affil)
    return


def configure(k, pts_path, centroids):
    """
    """
    try:
        vectors = np.loadtxt(pts_path, delimiter='\t')
        if centroids is None:
            for j in range(k):
                # centroids[i][j] = np.random.randint(min(expression[::][j]), max(expression[::][j]))
        else:
            centroids = np.loadtxt(pts_path, delimiter='\t')
    except IOError:
        raise IOError("Couldn't read in expression data and/or centroids file!")
    pass


def updateclusters(matrix, centroids_loc, pt_affil, centroid_posse):
    """
    """

    # return pt_affil, centroid_posse, settled
    pass


def updatecentroids(matrix, centroids_loc, centroid_posse):
    """
    """

    # return centroids_loc
    pass


def report(iterations, cluster_assignments):
    """
    """
    print("iterations: " + str(iterations))
    try:
        with file('./kmeans.out', 'w') as f:
            f.write('Allo, allo!')
            #for i in range(len(expression)):
            #    out.write(str(i) + '\t' + str(point_map[i] + 1) + '\n')
    except IOError:
        raise IOError('Trouble writing results to file!')
    return


# ------------ script execution --------------
if __name__ == '__main__':
    # check right number of args
    if len(sys.argv) >= 3 and len(sys.argv) <= 5:
        main(sys.argv[1:])
        exit(0)

    else:
        raise SyntaxError('Incorrect number of arguments!', sys.argv)

