from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np


def laplacian(A):
    # Calculate degree matrix D
    D = np.diag(np.sum(A, axis=1))

    # Calculate the inverse square root of D
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))

    # Calculate symmetric normalized Laplacian matrix
    L_sym = np.identity(A.shape[0]) - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)

    return L_sym


def spectral_clustering(affinity, k):
    # Compute Laplacian matrix
    L = laplacian(affinity)

    # Compute the first k eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = LA.eigh(L)
    idx = np.argsort(eigenvalues)[:k]
    k_eigenvectors = eigenvectors[:, idx]

    # Normalize rows of eigenvector matrix
    epsilon = 1e-12
    k_eigenvectors_norm = k_eigenvectors / (np.linalg.norm(k_eigenvectors, axis=1, keepdims=True) + epsilon)

    # Apply K-means clustering on the selected eigenvectors
    labels, _ = k_means_clustering(k_eigenvectors_norm, k)

    return labels

