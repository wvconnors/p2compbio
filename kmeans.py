# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:34:30 2015

@author: Will
"""

import numpy as np 
import sys
import os

#
# python kmeans.py k expression.dat max.it centroids.txt
# K-means input

expression = np.loadtxt(sys.argv[2], delimiter='\t') 
k = int(sys.argv[1]) 
max_it = int(sys.argv[3]) 

# K-means algorithm
#
# Initialize k centroids.
centroids = np.zeros((k, len(expression[0]))) 

# If a centroid file is provided, these are the starting centroids. If not, generate k centroids randomly (see details for initializing centroids below).
if (len(sys.argv) == 4):
    for i in range(k):
        for j in range(len(expression)):
            centroids[i][j] = np.random.randint(min(expression[::][j]), max(expression[::][j])) 
else:
    try:
        centroids = np.loadtxt(sys.argv[4], delimiter='\t') 
    except IndexError:
        print('centroids') 

iterations = 0 
converged = False 
old_centroids = centroids.copy() 
point_map = {} 
while (iterations <= max_it and not converged):
    cluster_map = {} 
    for i in range(k):
        cluster_map[i] = [] 

    # For each data point, assign it to a cluster such that the Euclidean distance from the data point to the centroid is minimized.
    for i in range(len(expression)):
        centr_dist = {} 
        for centroid in centroids:
            # compute distance to all centroids
            centr_dist[np.sqrt(np.sum((centroid - expression[i]) ** 2, dtype=np.int64))] = centroid 
        # assign point to sorted min centroid
        dist_list = np.sort(list(centr_dist.keys())) 
        # need more efficient way to determine centroid number...
        index = np.where(centroids == centr_dist[dist_list[0]]) 
        cluster_map[np.median(index[0])].append(expression[i]) 
        point_map[i] = np.median(index[0]) 

    # For each cluster, move the centroid to be at the center of all points that belong to that cluster.
    for i in range(k):
        new_centroid = centroids[i] 
        for dimen in range(len(cluster_map[i][0])):
            new_centroid[dimen] = np.average(cluster_map[i][::][dimen]) 
        centroids[i] = new_centroid 
    iterations += 1 
    if (old_centroids == centroids).all():
        converged = True 
    else:
        old_centroids = centroids.copy() 

# Iterate steps 2 and 3 until convergence or after a certain number of iterations (see details for stopping conditions below).
#
#
# K-means output
# iterations: 45
print("iterations: " + str(iterations)) 

# When your program reaches a stopping condition, cluster assignments should be written to a tab-delimited file called kmeans.out. The first column contains the genes listed by index as ordered in the input data file (starting at 1). The first gene must gene 1, the 2nd gene 2, etc. The second column should contain the number of the cluster the gene is assigned to.
with open("kmeans.out", 'w') as out:
    for i in range(len(expression)):
        out.write(str(i) + '\t' + str(point_map[i] + 1) + '\n') 
