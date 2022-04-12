# 2-d k-means

# import required libraries

import numpy as np
from pandas import DataFrame
from matplotlib import style
from matplotlib import pyplot as plt


# determine the euclidean distance between the data points

def euclidean_dist(a, b):
    return np.sqrt(sum(np.square(a - b)))


def measure_change(cluster_prev, cluster_new):  # this function helps us to know when to stop.
    res = 0
    for a, b in zip(cluster_prev, cluster_new):  # iterate over 2 arrays simultaneously
        res += euclidean_dist(a, b)  # calculate sum of their distances
    return res


def show_clusters(x, cluster, cluster_arr):
    df = DataFrame(dict(x=x[:, 0], y=x[:, 1], label=cluster))
    colors = {0: 'blue', 1: 'red'}  # show different data points in different colors
    fig, ax = plt.subplots(figsize=(8, 8))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.scatter(cluster_arr[:, 0], cluster_arr[:, 1], marker='*', s=150, c='green')
    plt.show()
    #print(plt.style.available)
    plt.style.use('seaborn')

def updateCentroids(x, k, clusterID):
    cluster_arr = []  # initialize array named cluster_arr
    for i in range(k):  # loop over number of clusters
        arr = []  # initialize array named arr
        for j in range(len(x)):  # loop over all the data points
            if clusterID[j] == i:  # if current point belongs to cluster, then append to the array "arr"
                arr.append(x[j])
        cluster_arr.append(np.mean(arr, axis=0))  # once we have all the points we will compute the mean and append to the array "cluster_arr"
    return np.asarray(cluster_arr)  # return cluster_arr


def assign_cluster(x, k, clustered_arrays):
    # clustered_arrays is the array containing centroids
    cluster = [-1] * len(
        x)  # declaring array named cluster with length x. Because, ith element of cluster represents ith data point
    for i in range(len(x)):  # loop over all the data points
        dist_arr = []  # initialize array named dist_arr
        for j in range(k):  # loop over the clusters
            dist_arr.append(euclidean_dist(x[i], clustered_arrays[j]))  # append the array with current distance of the cluster and current centroid
        idx = np.argmin(dist_arr)  # take the index of the minimum distanced data point so that we know which centroid is closer to data point
        cluster[i] = idx  # assign the idx value to current cluster array
    print(cluster)
    return np.asarray(cluster)  # return cluster array


def initialize_centroids(x, k):
    # initialize an array
    arr = []

    # run a loop for number of clusters(2 in our case)
    for i in range(k):
        # initialize random centroid for each cluster
        c1 = np.random.uniform(min(x[:, 0]), max(x[:, 0]))
        c2 = np.random.uniform(min(x[:, 1]), max(x[:, 1]))
        arr.append([c1, c2])
    # print(arr)
    # return array with 2 co-ordinates
    return np.asarray(arr)


def kmeans(x, k):
    cluster_prev = initialize_centroids(x, k)  # initialize centroids
    cluster = [0] * len(x)  # initialize an array
    cluster_change = 100  # used to determine the stopping criteria
    while cluster_change > .001:
        cluster = assign_cluster(x, k, cluster_prev)  # assign clusters
        show_clusters(x, cluster, cluster_prev)  # show clusters
        cluster_new = updateCentroids(x, k, cluster)  # update centroids
        cluster_change = measure_change(cluster_new, cluster_prev)  # measure change
        cluster_prev = cluster_new  # update previous cluster to new cluster


x = np.array([[2, 4], [1.7, 2.8], [7, 8], [8.6, 8], [3.4, 1.5], [9, 11]])  # given data points
k = 2  # number of clusters
kmeans(x, k)
