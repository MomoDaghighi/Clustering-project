from numba import jit, njit, prange
import numpy as np
from numpy import linalg as LA

from timeit import default_timer as timer
from spectral import spectral_clustering as spectral_clustering_old
from spectral import laplacian as laplacian_old
from kmeans import k_means_clustering as k_means_clustering_old
from mnist import construct_affinity_matrix as construct_affinity_matrix_old, \
    chamfer_distance as chamfer_distance_old, \
    rigid_transform as rigid_transform_old, \
    icp as icp_old

from metrics import clustering_score


# Numba optimized chamfer_distance function
@njit(fastmath=True)
def chamfer_distance(point_cloud1, point_cloud2):
    sum_dist_1_to_2 = 0.0
    sum_dist_2_to_1 = 0.0

    # Calculate distances from each point in point_cloud1 to the nearest point in point_cloud2
    for i in prange(point_cloud1.shape[0]):
        min_dist_1_to_2 = np.inf
        for j in prange(point_cloud2.shape[0]):
            dist = 0.0
            for k in prange(point_cloud1.shape[1]):
                diff = point_cloud1[i, k] - point_cloud2[j, k]
                dist += diff * diff
            if dist < min_dist_1_to_2:
                min_dist_1_to_2 = dist
        sum_dist_1_to_2 += np.sqrt(min_dist_1_to_2)

    # Calculate distances from each point in point_cloud2 to the nearest point in point_cloud1
    for i in prange(point_cloud2.shape[0]):
        min_dist_2_to_1 = np.inf
        for j in prange(point_cloud1.shape[0]):
            dist = 0.0
            for k in prange(point_cloud2.shape[1]):
                diff = point_cloud2[i, k] - point_cloud1[j, k]
                dist += diff * diff
            if dist < min_dist_2_to_1:
                min_dist_2_to_1 = dist
        sum_dist_2_to_1 += np.sqrt(min_dist_2_to_1)

    # Compute the Chamfer distance and normalize by the number of points
    return (sum_dist_1_to_2 / point_cloud1.shape[0] + sum_dist_2_to_1 / point_cloud2.shape[0]) * 0.5


# Numba optimized rigid_transform function
@njit
def custom_mean(array):
    sum = 0.0
    for value in array:
        sum += value
    return sum / len(array)


@njit
def custom_column_means(arr):
    means = np.empty(arr.shape[1])
    for i in range(arr.shape[1]):
        means[i] = custom_mean(arr[:, i])
    return means


@njit
def rigid_transform(A, B):
    assert A.shape == B.shape

    # Compute centroids using custom function
    centroid_A = custom_column_means(A)
    centroid_B = custom_column_means(B)

    # Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Construct cross-covariance matrix
    H = np.dot(A_centered.T, B_centered)

    # SVD decomposition - Numba will call out to the underlying LAPACK implementation for this step
    U, S, Vt = LA.svd(H)

    # Construct rotation matrix
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if LA.det(R) < 0:
        # Ensure Vt has the correct shape before modification
        if Vt.shape[0] > 2:
            Vt[2, :] *= -1
        else:
            Vt[:, -1] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate translation
    t = centroid_B - np.dot(R, centroid_A)

    return R, t


# Numba optimized icp function
@njit
def find_nearest_neighbor_indices(source, target):
    nearest_neighbor_indices = np.empty(source.shape[0], dtype=np.int64)
    for i in prange(source.shape[0]):
        min_dist = np.inf
        for j in range(target.shape[0]):
            dist = 0.0
            for k in prange(source.shape[1]):
                temp = source[i, k] - target[j, k]
                dist += temp * temp
            if dist < min_dist:
                min_dist = dist
                nearest_neighbor_indices[i] = j
    return nearest_neighbor_indices


@njit
def icp(source, target, max_iterations=100, tolerance=1e-5):
    prev_error = np.inf
    source = source.astype(np.float64)  # Ensure source is floating-point
    target = target.astype(np.float64)
    for i in range(max_iterations):
        # Find the nearest neighbors in the target for each point in the source
        nearest_neighbor_indices = find_nearest_neighbor_indices(source, target)
        closest_points = target[nearest_neighbor_indices]

        # Calculate rigid transformation
        R, t = rigid_transform(source, closest_points)

        # Ensure source and R have the same data type for matrix multiplication
        R = R.astype(source.dtype)

        # Apply transformation
        transformed_source = np.dot(source, R.T) + t

        # Calculate Chamfer distance if chamfer_distance is rewritten with Numba
        dist = chamfer_distance(transformed_source, target)

        # Check for convergence
        if np.abs(prev_error - dist) < tolerance:
            break
        prev_error = dist

    return transformed_source, R, t


# Numba optimized construct_affinity_matrix function
@njit(parallel=True)
def construct_affinity_matrix(point_clouds, max_iterations=100, tolerance=1e-5):
    num_point_clouds = len(point_clouds)
    affinity_matrix = np.zeros((num_point_clouds, num_point_clouds))
    epsilon = 1e-10  # A small constant to prevent division by zero or affinity becoming 1 due to rounding errors.
    point_clouds = point_clouds.astype(np.float64)
    # Iterate over pairs of point clouds
    for i in prange(num_point_clouds):
        for j in prange(i + 1, num_point_clouds):
            # Register point clouds with each other
            registered_cloud, _, _ = icp(point_clouds[i], point_clouds[j], max_iterations, tolerance)

            # Calculate symmetric Chamfer distance between registered clouds
            dist_ij = chamfer_distance(registered_cloud, point_clouds[j])
            dist_ji = chamfer_distance(point_clouds[j], registered_cloud)

            chamfer_dist = max((dist_ij + dist_ji) / 2.0, epsilon)

            # Convert distance to affinity
            affinity = np.exp(-chamfer_dist)

            # Fill in the symmetric affinity matrix
            affinity_matrix[i, j] = affinity
            affinity_matrix[j, i] = affinity

    # Fill diagonal with the highest affinity
    np.fill_diagonal(affinity_matrix, 1.0)

    return affinity_matrix


@njit(parallel=True)
def k_means_clustering(data, k, max_iterations=100):
    n_samples, n_features = data.shape
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = data[indices]

    for _ in range(max_iterations):
        # Compute distances between data points and centroids
        distances = np.empty((k, n_samples))
        for i in prange(k):
            for j in prange(n_samples):
                dist = 0
                for l in range(n_features):
                    dist += (data[j, l] - centroids[i, l]) ** 2
                distances[i, j] = np.sqrt(dist)

        # Assign labels based on closest centroid
        labels = np.argmin(distances, axis=0)

        # Compute new centroids
        new_centroids = np.empty_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for i in prange(n_samples):
            cluster_index = labels[i]
            counts[cluster_index] += 1
            for j in prange(n_features):
                new_centroids[cluster_index, j] += data[i, j]

        for i in prange(k):
            if counts[i] > 0:
                new_centroids[i] /= counts[i]

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


@njit(parallel=True)
def laplacian(A):
    size = A.shape[0]
    D = np.zeros((size, size))
    L_sym = np.zeros((size, size))

    # Calculate degree matrix D diagonals
    for i in prange(size):
        D[i, i] = np.sum(A[i, :])

    # Calculate the inverse square root of D and L_sym simultaneously
    for i in prange(size):
        for j in range(size):
            if D[i, i] != 0:
                D[i, i] = 1.0 / np.sqrt(D[i, i])
            L_sym[i, j] = A[i, j] * D[i, i] * D[j, j]

    return L_sym


@njit(parallel=True)
def spectral_clustering(affinity, k):
    # Compute the Laplacian matrix
    L = laplacian(affinity)

    # Compute the first k eigenvectors of the Laplacian matrix using the symmetric tridiagonal eigendecomposition
    eigenvalues, eigenvectors = LA.eigh(L)
    idx = np.argsort(eigenvalues)[:k]
    k_eigenvectors = eigenvectors[:, idx]

    # Normalize rows of eigenvector matrix
    epsilon = 1e-12
    k_eigenvectors_norm = np.empty_like(k_eigenvectors)

    for i in prange(k_eigenvectors.shape[0]):
        norm = np.sqrt(np.sum(k_eigenvectors[i, :] ** 2))
        if norm < epsilon:
            norm = epsilon
        for j in range(k_eigenvectors.shape[1]):
            k_eigenvectors_norm[i, j] = k_eigenvectors[i, j] / norm

    # Apply K-means clustering on the selected eigenvectors
    labels, _ = k_means_clustering(k_eigenvectors_norm, k)  # Retrieve only labels, ignore centroids

    return labels


# Performance comparison code
if __name__ == "__main__":
    # Load dataset
    dataset = "mnist"
    dataset2 = "moons"
    dataset = np.load(f"datasets/{dataset}.npz")
    dataset2 = np.load(f"datasets/{dataset2}.npz")
    X = dataset['data']  # feature points
    X2 = dataset2['data']
    y2 = dataset2['target']
    y = dataset['target']  # ground truth labels
    n_clusters = len(np.unique(y))  # number of clusters
    n_clusters_2 = len(np.unique(y2))
    point_cloud1 = X[0]
    point_cloud2 = X[1]
    A = X[0]
    B = X[1]
    source = X[0]
    target = X[1]
    data = X
    data2 = X2
    k = n_clusters
    k2 = n_clusters_2
    max_iterations = 100
    tolerance = 1e-5
    affinity_matrix = construct_affinity_matrix_old(X)
    affinity_matrix2 = construct_affinity_matrix(X)

    # Measure performance of each old function and its optimized version
    functions_to_test = [
        ('construct_affinity_matrix', construct_affinity_matrix_old, construct_affinity_matrix, [X]),
        ('spectral_clustering', spectral_clustering_old, spectral_clustering, [affinity_matrix, n_clusters]),
        ('chamfer_distance', chamfer_distance_old, chamfer_distance, [point_cloud1, point_cloud2]),
        ('rigid_transform', rigid_transform_old, rigid_transform, [A, B]),
        ('icp', icp_old, icp, [source, target, max_iterations, tolerance]),
        ('k_means_clustering', k_means_clustering_old, k_means_clustering, [data2, k2, max_iterations]),
        ('laplacian', laplacian_old, laplacian, [affinity_matrix])
    ]

    for func_name, old_func, new_func, args in functions_to_test:
        # Measure old function
        start_time_old = timer()
        old_result = old_func(*args)
        time_old = timer() - start_time_old

        # Measure new (optimized) function
        start_time_new = timer()
        new_result = new_func(*args)
        time_new = timer() - start_time_new

        # Print results
        print(f"{func_name} - Old implementation time: {time_old:.6f} seconds")
        print(f"{func_name} - New implementation time: {time_new:.6f} seconds")
        print()
    y_pred_old = spectral_clustering_old(affinity_matrix, n_clusters)
    y_pred_new = spectral_clustering(affinity_matrix2, n_clusters)

    print("Old Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred_old))
    print("New Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred_new))
