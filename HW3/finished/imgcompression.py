from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N (*3 for color images)
            S: min(N, D) * 1 (* 3 for color images)
            V: D * D (* 3 for color images)
        """
        N = X.shape[0]
        D = X.shape[1]
        if len(X.shape) == 2:
            U, S, V = np.linalg.svd(X)
        elif len(X.shape) == 3:
            U, S, V= np.zeros((N, N, 3)), np.zeros((np.minimum(N, D), 3)), np.zeros((D, D, 3))
            for i in range(3):
                U[:, :, i], S[:, i], V[:, :, i] = np.linalg.svd(X[:, :, i])
        return U, S, V

    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if len(U.shape) == 2:
            return U[:, :k] @ np.diag(S[0:k]) @ V[:k, :]
        elif len(U.shape) == 3:
            x = np.zeros((U.shape[0], V.shape[0], 3))
            for i in range(3):
                x[:, :, i] = U[:, :k, i] @ np.diag(S[:k, i]) @ V[:k, :, i]
            return x

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        return k * (X.shape[0] + X.shape[1] + 1) / (X.shape[0] * X.shape[1])

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        if len(S.shape) == 1:
            temp = 0
            for i in range(k):
                temp += S[i]**2
            return temp / np.sum(S**2)
        else:
            recovered_var = []
            for i in range(3):
                temp = 0
                for j in range(k):
                    temp += S[j, i]**2
                recovered_var.append(temp / np.sum(S[:, i]**2))
            return recovered_var
