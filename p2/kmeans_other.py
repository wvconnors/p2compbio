#!/usr/bin/env python
"""
Created on Sat Oct 24 19:34:30 2015 // Original
Last edited : Sept 20, 2016

unsupervised k-means gene clustering across normalized multi-experiment TSV gene expression data
@author: Will Connors
"""

import numpy as np;
import sys;


# python kmeans.py k expression.dat max.it centroids.txt
# K-means inputs
def main(k, datain, cut_off, centrin=None):
    """
    Specification wrapper for kmeans algorithm function

    parameters:

    [required]
    k  :: integer number of groups/centroids to run on the data

    datain :: the datafile location [valid class spec format TSV required -
            no missing data values, 2d MxN normalized integer array]

    cut_off :: an integer number of algorithmmic iterations to cap execution for non converging runs

    [optional]
    centrin :: file location of pregen centroid locations .txt [valid class spec -
            must have at least k rows of valid TSV floats matching data dim,
            first k entries used]. defaults to random, bounded by expression min-max

    output:
    Iterations reported to terminal [style spec in function]
    writes a working directory file named 'kmeans.out', a simple TSV-
    Lists gene number (data row number) TAB gene's cluster assignment (ordered 1-k)
    """

    """------------------------PROGRAM START--------------------------------"""

    # Data filein
    try:
        gene_data = np.loadtxt(datain, delimiter='\t', dtype=np.float32);
    except Exception:
        print "gene_data argument read error!";
        raise Exception;

    # Centroid config
    k = int(k);
    if centrin is None:
        # generate random[linear probability over data min-max by dim]:
        datadims = len(gene_data[0]);
        centroids = np.zeros((k, datadims), dtype=np.float32);
        np.random.seed();

        for dim in range(datadims):
            # scale random [0,1) to size and range of values within dim/experiment
            centroids[:, dim] = (max(gene_data[:, dim]) - min(gene_data[:, dim])) * \
                                np.random.ranf((1, k)) + min(gene_data[:, dim]);

    else:
        # read in first k centroids from file
        try:
            centroids = np.loadtxt(centrin, delimiter='\t', dtype=np.float32);
            centroids = centroids[0:k];  # slicing not inclusive!
        except OSError:
            print "centroids.txt arguement error! 404, invalid format, or <k rows";
            raise OSError;

    # MEAT AND POTATOES
    # Execute algorithm with given conditions, and report results
    dataOut(kmeansAlgo(gene_data, centroids, cut_off));


def kmeansAlgo(nppoints, npcentr, ubound):
    """#K-means algorithm, plug and chug!
   nppoints :: an np array of M points by N dimensions
   npcentr :: an np array of k centroids by N dimensions // K derived from length in dim 0 (rows)
   ubound :: point of cut off if not converged

   returns list of tab seperated gene/point final cluster assignments
    """

    iterations = 0;
    converged = False;

    # generate a list of ptobj's, with pt getting its gene# and coords
    ptobjs = [pointObj(coords, num) for num, coords in enumerate(nppoints, start=1)];

    while (iterations < ubound):
        converged = True;  # Until proven otherwise by a pt switching clusters

        # assign all points to clusters
        for pt in ptobjs:
            # create a list of all centr distances and a map from the distance to the centr ID
            centr_dists = {float(np.sqrt(sum(np.subtract(pt.querycoords(), centrcoords) ** 2))): y for y,
                                                                                                       centrcoords in
                           enumerate(npcentr, start=1)};
            min_dist = min(centr_dists.keys());
            # assign the cluster and get the changed? boolean from the pt obj
            changed = pt.assigncl(centr_dists[min_dist]);
            if converged and changed:
                converged = False;

        iterations += 1;

        if converged:
            break

        else:
            # For each cluster, move the centroid to be at the center of all points that belong to that cluster.
            for clust in range(len(npcentr)):
                # build mask of pts that belong to cluster
                maskbuild = np.zeros_like(nppoints, dtype=np.bool_);
                for x, pt in enumerate(ptobjs):
                    if pt.querycl() == clust + 1:
                        maskbuild[x, :] = True;
                masked = nppoints[maskbuild];
                if maskbuild.any():
                    # conditional debugs empty centroids
                    # average along each dimension(experiment) of the pts
                    npcentr[clust, :] = masked.mean(axis=0);

    # once converged or cut off, return its and point assignments
    final_pt_clusters = [str(x) for x in ptobjs];

    return (iterations, final_pt_clusters)


def dataOut((its, gene_assignment)):
    #
    #
    # K-means output style
    # iterations: 45
    print("iterations: " + str(its));

    with open("kmeans.out", 'w') as out:
        for i in gene_assignment:
            out.write(i + '\n');


class pointObj:
    """Helper class for kmeans, defines a point object with coordinates,
    a possible cluster assignment, and a possible name.

    """

    def __init__(self, location, name=None):
        self.cluster = None;
        self.coords = location;
        self.name = name;

    def assigncl(self, clustnum):
        """ Returns 1 if it changed, 0 if not"""
        changed = False;
        if (self.cluster != clustnum):
            changed = True;
        self.cluster = clustnum;
        return changed

    def querycl(self):
        return self.cluster

    def querycoords(self):
        return self.coords

    def queryname(self):
        return self.name

    def __str__(self):
        return str(self.name) + '\t' + str(int(self.cluster))


############################### stand alone script execution ###############################################
if __name__ == "__main__" and len(sys.argv) == 5:
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]);
elif __name__ == "__main__" and len(sys.argv) == 4:
    main(sys.argv[1], sys.argv[2], sys.argv[3]);
elif __name__ == "__main__":
    print('script arguement error! Did you put in the 3 mandatory args correctly?');
    raise BaseException;
