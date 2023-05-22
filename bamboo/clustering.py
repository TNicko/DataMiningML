
# ----------------------------------------------------------------------------------------------------
# Description: Contains functions for working with clusters and cluster evaluation with Sklearn.
# ----------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def kmeans_elbow(X: np.ndarray, num_clusters: int, seed: int | None = None) -> list:
    """
    Compute the sum of squared errors (SSE) for a range of number of clusters in KMeans. Used to aid in 
    choosing the optimal number of clusters (the 'elbow point' in the plot of SSE vs number of clusters).

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


def kmeans_silhouette(X: np.ndarray, kmeans: KMeans, num_clusters: int, seed: int | None = None) -> list:
    """
    Calculates silhouette scores for KMeans clustering with different numbers of clusters.

    Parameters:
    -----------
    X : np.ndarray
        The input data (features), where each row is a sample and each column is a feature.
    num_clusters : int
        The maximum number of clusters to consider.
    seed : int | None, optional
        The random seed to use for initializing the KMeans algorithm.
    Returns:
    --------
    silhouette_scores : list
        A list of silhouette scores for different numbers of clusters from 2 to num_clusters.
    """
    silhouette_scores = []
    for k in range (2, num_clusters+1):
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        kmeans.fit(X)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return silhouette_scores


def fit_kmean_cluster(X: np.ndarray, dreduce: object, n_clusters: int, seed: int | None = None) -> KMeans:
    """
    Fits a KMeans clustering model to the feature data whilst applying dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Input feature array.
    dreduce : object
        Dimensionality reduction object e.g. (PCA or TSNE).
    n_clusters : int
        Number of clusters to create.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Fitted KMeans object and the transformed data array.
    """
    # Initialise KMeans object
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init="k-means++", 
        n_init='auto',
        n_init=50,
        max_iter=500, 
        random_state=seed)

    # Fit and transform data
    X_reduce = dreduce.fit_transform(X)
    kmeans.fit(X_reduce)

    return kmeans, X_reduce


def get_cluster_data(kmeans: KMeans, X_reduce: np.ndarray, y: np.ndarray = None) -> dict:
    """
    Retrieves cluster-related data from the KMeans model.

    Parameters
    ----------
    kmeans : KMeans
        Fitted KMeans clustering model.
    X_reduce : np.ndarray
        Transformed data array.
    y : np.ndarray, optional
        True labels for evaluating the clustering performance.

    Returns
    -------
    dict
        Dictionary containing cluster labels, silhouette score, and optionally ARI score.
    """
    try:
        labels = kmeans.labels_
    except:
        print("Error: No labels found for KMeans object.")
        return None
    
    silhoutte = silhouette_score(X_reduce, labels)
    
    data = {
        "labels": labels, 
        "silhouette": silhoutte, 
    }
    # If y is provided, calculate ARI
    if y is not None:
        ari = adjusted_rand_score(y, labels)
        data["ari"] = ari

    return data



