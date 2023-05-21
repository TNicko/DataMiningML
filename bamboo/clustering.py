
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def kmeans_elbow(X: np.ndarray, num_clusters: int, seed: int | None = None) -> list:
    """
    Compute the sum of squared errors (SSE) for a range of number of clusters in KMeans. This method is often used
    to aid in choosing the optimal number of clusters (the 'elbow point' in the plot of SSE vs number of clusters).

    Parameters
    ----------
    X : np.ndarray
        The input data (features), where each row is a sample and each column is a feature.
    num_clusters : int
        The maximum number of clusters to consider.
    seed : int, None, optional
        The random seed to use for initializing the KMeans algorithm.

    Returns
    -------
    list
        A list of the SSE for each number of clusters from 1 to num_clusters.
    """
    sses = [] # list of sum of squared errors for each k
    for k in range (1, num_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        kmeans.fit(X)
        sses.append(kmeans.inertia_)

    return sses

def kmeans_silhouette(X: np.ndarray, num_clusters: int, seed: int | None = None) -> list:
    silhouette_scores = []
    for k in range (2, num_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return silhouette_scores