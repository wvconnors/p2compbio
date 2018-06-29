#!/usr/bin/env python
"""  kmeans.py is a python 2.7 script implementing kmeans clustering, per the
spec of Stanford class BMI 214, project 2
@author: Will Connors
usage:
  python kmeans.py k expression.dat max.it centroids.txt
    INPUTS:
    k :: int; [1+]; the number of centroids / groupings to use
    data_file :: string; file path to data source TSV where rows are samples
      and columns are dimensions
    max.it :: int; [1+] cutoff for maximum iterations if run does not converge
    optional: centroids.txt :: string; file path to premade centroid starting
      coordinates, with at least k rows
    OUTPUTS:
    No returns, but writes to [WORKING_DIR]/kmeans.out the clusters that every
     sample finished in (both centroids and points 1-indexed)
"""

from __future__ import absolute_import, division, print_function
import sys
import numpy as np


def main(k, data_file, max_it, centroids_file = None):
    """ function main() encapsulates the entire functionality and execution
     of kmeans.py, implementing kmeans clustering of TSV texts.
    INPUTS:
    k :: int; [1+]; the number of centroids / groupings to use
    data_file :: string; file path to data source TSV where rows are samples
      and columns are dimensions
    max_it :: int; [1+] cutoff for maximum iterations if run does not converge
    optional: centroids_file :: string; file path to premade centroid starting
      coordinates, with at least k rows
    OUTPUTS:
    No returns, but writes to [WORKING_DIR]/kmeans.out the clusters that every
     sample finished in (both centroids and points 1-indexed)
    """
    k = int(k)
    max_it = int(max_it)
    matrix, centroid_loc = configure(k, data_file, centroids_file)
    num_pts, num_dims = matrix.shape
    # efficient update require require dual datamapping structures:
    #  pt ID -> centroid ID
    pt_affil = []
    # centroid ID-> list of cluster member point IDs
    centroid_posse = {}
    # centroid_posse[0] is Null centroid
    for cen in range(k+1):
        centroid_posse[cen] = []
    for pt in range(num_pts+1):   #  0 - 78
        centroid_posse[0].append(pt)
        pt_affil.append(int(0))
    converged_after = max_it
    # iteration 0 pt assignment:
    pt_affil, centroid_posse, settled = updateclusters(matrix, centroid_loc,
                                                       pt_affil, centroid_posse)

    for w in range(1, max_it+1):
        centroid_loc = updatecentroids(matrix, centroid_loc, centroid_posse)
        pt_affil, centroid_posse, settled = updateclusters(matrix, centroid_loc,
                                                           pt_affil, centroid_posse)
        if settled:
            converged_after = w
            break

    # else:
    #     print('Warning, max iterations reached without stable assignments!')

    report(converged_after, pt_affil)
    return


def configure(k, pts_path, centroids):
    """ function configure(), called by main(), handles file read in, centroid read or generation,
    and related exception handling
    INPUTS:
      k :: int; the number of centroids / groupings to use
      pts_path :: string; path to TSV of data to cluster
      centroids :: None | string; None or path to pre-made centroid starting coords
    OUTPUTS:
      vectors :: np.ndarray<float>; data from pts_path as numpy array
      centr_0 :: np.ndarray<float>; starting location of centroids
    """

    try:
        vectors = np.loadtxt(pts_path, delimiter='\t')
        num_pts, num_dims = vectors.shape
        centr_0 = np.zeros((k, num_dims))
        if centroids is None:
            for i in range(k):
                for j in range(num_dims):
                    known_min = min(vectors[:, j])
                    known_max = max(vectors[:, j])
                    # not the most complex solution, but suitable
                    centr_0[i, j] = np.random.uniform(known_min, known_max)

        else:
            centr_0 = np.loadtxt(centroids, delimiter='\t')
    except IOError:
        raise IOError("Couldn't read in expression data and/or centroids file!")
    # file might have more than k centroids
    if len(centr_0) > k:
        centr_0 = centr_0[:k]
    return vectors, centr_0


def updateclusters(matrix, centroid_loc, pt_affil, centroid_posse):
    """ function updateclusters(), called by main(), in alternating steps of kmeans() convergence
    INPUTS:
        matrix ::
        centroids_loc ::
        pt_affil ::
        centroid_posse ::
    OUTPUTS:
        pt_affil ::
        centroid_posse ::
        settled ::
    """

    num_pts, num_dims = matrix.shape
    no_delta = True

    # assign points to nearest centroid
    for pt in range(1,num_pts+1):

        # last match 1 indexed id
        last_match = pt_affil[pt]
        # find distance to all centroids, sort, return label
        # note combining float and int/ids requires conversion later
        expanded_mat = np.zeros_like(centroid_loc)
        expanded_mat += matrix[pt-1]
        dist = np.linalg.norm(expanded_mat - centroid_loc, axis=1)
    #         dist = [(it, np.linalg.norm(matrix[pt, :] - centr))
    #                 for it, centr in enumerate(centroid_loc, start=1)]
    #        dist = np.asarray(dist)
        # sort just the distance column, and use it as index to resort
        # best_match = dist[dist[:, 1].argsort()]
        # centr_id via enumerate() 1-indexed
        centr_id = 1 + int(dist.argsort()[0])
        if centr_id != last_match:
            no_delta = False
            pt_affil[pt] = centr_id
            # centroid_posse 0-indexed, [0-k]
            centroid_posse[centr_id].append(pt)
            centroid_posse[last_match].remove(pt)

    return pt_affil, centroid_posse, no_delta


def updatecentroids(matrix, locations, centroid_posse):
    """ function updatecentroids(), called by main(), in alternating steps of kmeans() convergence
    matrix ::

    centroid_posse ::

    centroids_loc ::
    """

    num_pts, num_dims = matrix.shape
    num_centr = len(centroid_posse) # centroids in posse are +1 for ghost bucket
    for centroid in range(1, int(num_centr)):
        posse = np.array(centroid_posse[centroid], dtype=int)  # element-wise subtract from pt_ids to 0-index
        if len(posse) >= 1:
            posse -= 1
            my_voters = matrix[posse.tolist()]
            for dim in range(int(num_dims)):
                locations[centroid-1, dim] = float(np.sum(my_voters[:, dim])) / float(posse.size)

    return locations


def report(iterations, cluster_assignments):
    """ function report(), called by main(), handles post-UX, file reading, and error handling
    iterations :: int;
    cluster_assignments :: list(<int>);
    """
    print("iterations: " + str(iterations))
    try:
        with open('./kmeans.out', 'w') as f:
            results = tuple(enumerate(cluster_assignments, start=1))
            for ptid, cluster in results:
                f.write('%s \t %s \n' % (ptid, cluster))
    except IOError:
        raise IOError('Trouble writing results to file!')
    return


# ------------ script execution --------------
if __name__ == '__main__':
    # check right number of args
    if len(sys.argv) >= 3 and len(sys.argv) <= 5:
        main(*sys.argv[1:])
        exit(0)

    else:
        raise SyntaxError('Incorrect number of arguments!', sys.argv)
