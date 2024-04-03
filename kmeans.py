import numpy as np


def k_means_clustering(data, k, max_iterations=100):
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iterations):
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([data[labels == i].mean(axis=0) if i in labels else centroids[i] for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids
