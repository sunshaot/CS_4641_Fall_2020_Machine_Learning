
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)


class KMeans(object):

    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [5 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
                """
        x = np.array(x)
        y = np.array(y)
        x_prime = x[:,np.newaxis,:]
        y_prime = y
        dist_temp = x_prime - y_prime
        dist = np.linalg.norm(dist_temp, axis = -1)
        return dist

    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        random = np.random.choice(len(points), K, replace = False)
        centers = []
        for i in random:
            centers.append(points[i])
        centers = np.array(centers)
        return centers

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        dist = KMeans().pairwise_dist(centers, points)
        cluster_idx = np.argmin(dist, axis = 0)
        return cluster_idx

    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """
        old_centers = np.array(old_centers)
        K, D = old_centers.shape
        centers = np.empty((K, D))
        for i in range(K):
            cluster = points[np.argwhere(cluster_idx == i)]
            centers[i] = np.mean(cluster, axis = 0)
        return centers

    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        loss = np.sum(np.linalg.norm(points - centers[cluster_idx]) ** 2)
        return loss

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):  # [10 pts]
        """Plots loss values for different number of clusters in K-Means

        Args:
            image: input image of shape(H, W, 3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        """
        loss_array = np.empty(max_K)
        data = np.array(data)
        for i in range(max_K):
            _, _, loss_array[i] = KMeans()(data, i + 1)
        return loss_array

def intra_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster

    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """
    list_same_cluster = data[np.argwhere(labels == cluster_idx)]
    intra_dist_cluster = np.empty(len(list_same_cluster))
    for i in range(len(list_same_cluster)):
        total_dist = 0
        for j in range(len(list_same_cluster)):
            total_dist += np.linalg.norm(list_same_cluster[i] - list_same_cluster[j])
        intra_dist_cluster[i] = total_dist / (len(list_same_cluster) - 1)
    return intra_dist_cluster

def inter_cluster_dist(cluster_idx, data, labels):  # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """
    list_same_cluster = np.nan_to_num(data[np.argwhere(labels == cluster_idx)])
    list_notsame_cluster = np.nan_to_num(data[np.argwhere(labels != cluster_idx)])
    inter_dist_cluster = np.empty(len(list_same_cluster))
    for i in range(len(list_same_cluster)):
        min_table = np.empty(len(np.unique(labels)) - 1)
        count = 0
        for l in np.unique(labels):
            if l == cluster_idx:
                continue
            temp_cluster = data[np.argwhere(labels == l)]
            total_dist = 0
            for j in range(len(temp_cluster)):
                total_dist += np.linalg.norm(list_same_cluster[i] - temp_cluster[j])
            avg_dist = total_dist / len(temp_cluster)
            min_table[count] = avg_dist
            count += 1
        inter_dist_cluster[i] = np.amin(min_table)
    return inter_dist_cluster


def silhouette_coefficient(data, labels):  # [2 pts]
    """
    Finds the silhouette coefficient of the current cluster assignment

    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        silhouette_coefficient: Silhouette coefficient of the current cluster assignment
    """
    sum_total = 0
    temp_list = []
    unique = np.unique(labels)
    for cluster_idx in unique:
        if len(data[np.argwhere(labels == cluster_idx)]) == 1:
            temp_list.append(0)
        else:
            inter_dist_cluster = np.nan_to_num(inter_cluster_dist(cluster_idx, data, labels))
            intra_dist_cluster = np.nan_to_num(intra_cluster_dist(cluster_idx, data, labels))
            result_list = (inter_dist_cluster - intra_dist_cluster)/np.maximum(inter_dist_cluster, intra_dist_cluster)
            sum_result = np.sum(result_list)
            temp_list.append(sum_result)
    return np.sum(np.array(temp_list) / data.shape[0])