import numpy as np
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score


def construct_affinity_matrix(data, affinity_type, *, k=9, sigma=0.1):
    n_samples = data.shape[0]

    # Compute pairwise distances
    sq_dists = np.sum((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2, axis=-1)

    if affinity_type == 'knn':
        # Find k nearest neighbors for each point
        # Initialize the affinity matrix with zeros
        affinity_matrix = np.zeros((n_samples, n_samples))

        # For each sample, sort the distances and get the indices of the k closest points
        sorted_indices = np.argsort(sq_dists, axis=1)
        for i in range(n_samples):
            # Set the corresponding entries in the affinity matrix to 1
            affinity_matrix[i, sorted_indices[i, 1:k + 1]] = 1

        # Make the matrix symmetric
        affinity_matrix = np.maximum(affinity_matrix, affinity_matrix.T)

    elif affinity_type == 'rbf':
        # Apply RBF kernel
        affinity_matrix = np.exp(-sq_dists / (2. * sigma ** 2))

    else:
        raise ValueError("Invalid affinity matrix type")

    return affinity_matrix


def plot_clusters(X, y, title, ax):
    # Plot the clusters on the given Axes object
    unique_labels = np.unique(y)
    for label in unique_labels:
        ax.scatter(X[y == label, 0], X[y == label, 1], label=f"Cluster {label}")
    ax.set_title(title)
    ax.legend()


if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

    # Iterate over datasets
    for i, ds_name in enumerate(datasets):
        dataset = np.load(f"datasets/{ds_name}.npz")
        X = dataset['data']  # feature points
        y = dataset['target']  # ground truth labels
        n = len(np.unique(y))  # number of clusters

        # Run clustering algorithms
        k = 9
        sigma = 0.1

        y_km, _ = k_means_clustering(X, n)
        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)
        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        # Plot the results
        plot_clusters(X, y, f"Ground Truth ({ds_name})", axes[i, 0])
        plot_clusters(X, y_km, f"K-means ({ds_name})", axes[i, 1])
        plot_clusters(X, y_rbf, f"RBF + Spectral ({ds_name})", axes[i, 2])
        plot_clusters(X, y_knn, f"KNN + Spectral ({ds_name})", axes[i, 3])

        # Print the clustering scores for each algorithm
        print(f"K-means on {ds_name}: {clustering_score(y, y_km)}")
        print(f"RBF affinity on {ds_name}: {clustering_score(y, y_rbf)}")
        print(f"KNN affinity on {ds_name}: {clustering_score(y, y_knn)}")

    # Adjust layout
    plt.tight_layout()
    plt.show()
