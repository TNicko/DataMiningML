# ----------------------------------------------------------------------------------------------------
# Description: Contains functions for preparing processed dataset for training & testing.
# ----------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


def normalize_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Normalizes the training and testing datasets using a MinMaxScaler.
    
    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    X_test : np.ndarray
        The testing data.

    Returns
    -------
    tuple
        The normalized training and testing data.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


def get_const_features(X_train: np.ndarray, X_names: list, threshold: float = 0.01) -> list:
    """
    Identifies near constant features (columns) in the training data.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    X_names : list
        The names of the features (columns) in the data.
    threshold : float, optional
        The variance threshold for identifying constant features. Default is 0.01.

    Returns
    -------
    list
        The selected and constant column names.
    """
    # Removing constant features
    constant_filter = VarianceThreshold(threshold=threshold)

    data_constant = constant_filter.fit_transform(X_train)

    # Get all non constant columns
    selected_indices = constant_filter.get_support(indices=True)
    selected_columns = [X_names[i] for i in selected_indices]

    # Get constant columns
    constant_columns = [column for column in X_names if column not in selected_columns]

    return selected_columns, constant_columns


def get_feature_correlations(X_train: np.ndarray, y_train: np.ndarray, X_names) -> dict:
    """
    Calculates the correlation between each input feature and the target variable.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training targets.
    X_names : list
        The names of the features (columns) in the data.

    Returns
    -------
    dict
        A dictionary mapping each feature to its correlation with the target variable.
    """
    # Calculate the correlation between each input feature and the target variable
    correlations = np.corrcoef(X_train, y_train, rowvar=False)[-1, :-1]

    # Create a dictionary to store the correlation of each feature with the target variable
    feature_correlations = {feature: corr for feature, corr in zip(X_names, correlations)}

    return feature_correlations


def get_low_correlation_features(feature_correlations: dict, threshold: float = 0.05) -> list:
    """
    Identifies features that have a correlation with the target variable below a given threshold.

    Parameters
    ----------
    feature_correlations : dict
        A dictionary mapping each feature to its correlation with the target variable.
    threshold : float, optional
        The correlation threshold for identifying low-correlation features. Default is 0.05.

    Returns
    -------
    list
        The names of the low-correlation features.
    """
    low_correlation_features = [feature for feature, corr in feature_correlations.items() if abs(corr) < threshold or np.isnan(corr)]
    return low_correlation_features


def plot_feature_correlations(feature_correlations: dict):
    """
    Creates a bar plot to visualize the correlation between input features and the target variable.

    Parameters
    ----------
    feature_correlations : dict
        A dictionary mapping each feature to its correlation with the target variable.
    """
    plt.figure(figsize=(12, 6))
    corr_values = np.array(list(feature_correlations.values()))
    x_labels = list(feature_correlations.keys())
    nan_indices = np.where(np.isnan(corr_values))[0]
    nan_labels = [x_labels[i] for i in nan_indices]
    print(nan_labels)
    plt.bar(x_labels, corr_values)
    for i in nan_indices:
        plt.plot([i-0.4, i+0.4], [0, 0], color='r', linewidth=1)
    plt.xlabel('Features')
    plt.ylabel('Correlation with Target Variable')
    plt.title('Feature Correlations with the Target Variable')
    plt.xticks(rotation=45)
    if len(nan_labels) > 0:
        plt.legend(['NaN Correlation'])
    plt.show()