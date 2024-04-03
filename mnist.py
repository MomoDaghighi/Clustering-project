import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from spectral import spectral_clustering
from metrics import clustering_score


def chamfer_distance(point_cloud1, point_cloud2):
    # Calculate distances from each point in point_cloud1 to all points in point_cloud2
    distances_1_to_2 = LA.norm(point_cloud2[:, None] - point_cloud1, axis=-1)
    sum_dist_1_to_2 = np.min(distances_1_to_2, axis=1).sum()

    # Calculate distances from each point in point_cloud2 to all points in point_cloud1
    distances_2_to_1 = LA.norm(point_cloud1[:, None] - point_cloud2, axis=-1)
    sum_dist_2_to_1 = np.min(distances_2_to_1, axis=1).sum()

    # Compute the Chamfer distance
    dist = (sum_dist_1_to_2 / len(point_cloud1)) + (sum_dist_2_to_1 / len(point_cloud2))
    return dist / 2


def rigid_transform(A, B):
    assert A.shape == B.shape

    # Subtract centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Construct cross-covariance matrix
    H = A_centered.T @ B_centered

    # SVD decomposition
    U, S, Vt = LA.svd(H)

    # Calculate rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation matrix with a determinant of 1
    if LA.det(R) < 0:
        # Ensure Vt has the correct shape before modification
        if Vt.shape[0] > 2:
            Vt[2, :] *= -1
        else:
            Vt[:, -1] *= -1
        R = Vt.T @ U.T

        # Calculate translation
    t = centroid_B - (R @ centroid_A)

    return R, t


def icp(source, target, max_iterations=100, tolerance=1e-5):
    prev_error = float('inf')

    for i in range(max_iterations):
        # Find the nearest neighbors in the target for each point in the source
        distances = np.sqrt(((target[None, :] - source[:, None]) ** 2).sum(-1))
        nearest_neighbor_indices = np.argmin(distances, axis=1)
        closest_points = target[nearest_neighbor_indices]

        # Calculate rigid transformation
        R, t = rigid_transform(source, closest_points)

        # Apply transformation
        transformed_source = (R @ source.T).T + t.T

        # Calculate Chamfer distance
        dist = chamfer_distance(transformed_source, target)

        # Check for convergence
        if np.abs(prev_error - dist) < tolerance:
            break
        prev_error = dist

    return transformed_source, R, t


def construct_affinity_matrix(point_clouds):
    num_point_clouds = len(point_clouds)
    affinity_matrix = np.zeros((num_point_clouds, num_point_clouds))
    epsilon = 1e-10  # A small constant to prevent division by zero or affinity becoming 1 due to rounding errors.
    # Iterate over pairs of point clouds
    for i in range(num_point_clouds):
        for j in range(i + 1, num_point_clouds):
            # Register point clouds with each other
            registered_cloud, _, _ = icp(point_clouds[i], point_clouds[j])

            # Calculate symmetric Chamfer distance between registered clouds
            dist_ij = chamfer_distance(registered_cloud, point_clouds[j])
            dist_ji = chamfer_distance(point_clouds[j], registered_cloud)

            chamfer_dist = max((dist_ij + dist_ji) / 2.0, epsilon)

            # Convert distance to affinity. For example, using the exponential function to ensure stability
            affinity = np.exp(-chamfer_dist)

            # Fill in the symmetric affinity matrix
            affinity_matrix[i, j] = affinity
            affinity_matrix[j, i] = affinity

    # Fill diagonal with the highest affinity
    np.fill_diagonal(affinity_matrix, 1.0)

    return affinity_matrix


if __name__ == "__main__":
    dataset = "mnist"
    dataset = np.load(f"datasets/{dataset}.npz")
    X = dataset['data']  # feature points
    y = dataset['target']  # ground truth labels
    n_clusters = len(np.unique(y))  # number of clusters

    # Construct affinity matrix
    Ach = construct_affinity_matrix(X)

    # Perform spectral clustering
    y_pred = spectral_clustering(Ach, n_clusters)

    # Calculate and print clustering score
    score = clustering_score(y, y_pred)
    print(f"Chamfer affinity on {dataset}: {score}")

    # Compute the first 3 eigenvectors of the affinity matrix
    eigenvalues, eigenvectors = LA.eigh(Ach)
    idx = np.argsort(eigenvalues)[::-1][:3]  # Indices of the first 3 largest eigenvalues
    first_3_eigenvectors = eigenvectors[:, idx]

    # Visualize the affinity matrix in 3D using the first 3 eigenvectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the first 3 eigenvectors, color by cluster label
    scatter = ax.scatter(first_3_eigenvectors[:, 0], first_3_eigenvectors[:, 1], first_3_eigenvectors[:, 2],
                         c=y_pred, cmap='viridis', s=15)

    # Create a legend and title
    ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set_title('3D Visualization using First 3 Eigenvectors')

    # Label the axes
    ax.set_xlabel('Eigenvector 1')
    ax.set_ylabel('Eigenvector 2')
    ax.set_zlabel('Eigenvector 3')

    # Show the plot
    plt.show()
