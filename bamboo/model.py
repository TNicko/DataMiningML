import joblib
import os
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc, det_curve
from sklearn.experimental import enable_halving_search_cv, enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

class ModelManager:
    """
    A utility class to handle the training, prediction, and result analysis of a machine learning model.
    This class can handle both basic model fitting and hyperparameter tuning using Grid Search or Halving Grid Search.

    Attributes
    ----------
    model : object
        The underlying model object.
    params : dict
        A dictionary of parameters for hyperparameter tuning. Keys are parameter names, values are lists of possible parameter values.
    search : object
        A Scikit-learn search object (GridSearchCV or HalvingGridSearchCV).
    search_type : str
        The type of search to use for hyperparameter tuning. Options are "GridSearch" or "HalvingGridSearch".

    Methods
    -------
    fit(X, y):
        Fits the model to the provided data. If params is provided, performs hyperparameter tuning using the specified search_type.
    predict(X):
        Returns the model's predictions on the provided data.
    save_model(path):
        Saves the trained model to the specified path.
    save_results(path):
        Saves the results of the hyperparameter tuning to the specified path.
    load_model(path):
        Loads a model from the specified path.
    load_results(path):
        Loads search results from the specified path.
    get_fit_results():
        Returns the results of the hyperparameter tuning.
    get_best_configuration():
        Returns a summary of the best configuration found by the hyperparameter tuning.
    get_configurations(sort_by='Score', n_results=None, ascending=False):
        Returns a list of configuration summaries, sorted by the specified criterion.
    """
    def __init__(self, model: object = None, params: dict = None, search_type: str = "GridSearch"):
        self.model = model
        self.params = params
        self.search = None
        self.search_type = search_type

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.params:
            if self.search_type == "GridSearch":
                self.search = GridSearchCV(self.model, self.params, cv=5, verbose=2)
            elif self.search_type == "HalvingGridSearch":
                self.search = HalvingGridSearchCV(self.model, self.params, cv=5, verbose=2)
            else:
                raise ValueError(f"Unknown search_type: {self.search_type}")

            self.search.fit(X, y)
            self.model = self.search.best_estimator_

        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model:
            return self.model.predict(X)
        else:
            raise Exception("Model not trained yet!")
        
    def save_model(self, path: str):
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
        else:
            raise Exception("Model not trained yet!")
        
    def save_results(self, path: str):
        if self.search:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.search, path)
        else:
            raise Exception("Model trained without parameters grid, no results available!")

    def load_model(self, path: str):
        self.model = joblib.load(path)
    
    def load_results(self, path: str):
        self.search = joblib.load(path)

    def get_fit_results(self) -> dict:
        if self.search:
            return self.search.cv_results_
        else:
            raise Exception("Model trained without parameters grid, no fit results available!")

    def get_best_configuration(self) -> dict:
        if self.search:
            best_index = self.search.best_index_
            best_fit_time = self.search.cv_results_['mean_fit_time'][best_index]

            return {
                'Best Params': self.search.best_params_,
                'Best Score': round(self.search.best_score_, 2),
                'Time Taken': round(best_fit_time, 3),
            }
        else:
            raise Exception("Model trained without parameters grid, no summary available!")

    def get_configurations(self, sort_by: str = 'Score', n_results: int = None, ascending: bool = False) -> list[dict]:
        if self.search:
            results = self.get_fit_results()

            # Create a list of dictionaries, each containing details of one configuration.
            config_details = []
            for params, score, time in zip(results['params'], results['mean_test_score'], results['mean_fit_time']):
                config_details.append({
                    'Params': params,
                    'Score': score,
                    'Time Taken': time
                })

            # Check if sorting criterion is valid
            if sort_by not in ['Score', 'Time Taken']:
                raise ValueError("Invalid sort_by value. Choose between 'Score' and 'Time Taken'")

            # Sort the configurations
            sorted_configurations = sorted(config_details, key=lambda x: x[sort_by], reverse=not ascending)

            # Return the specified number of results, or all if n_results is None.
            return sorted_configurations[:n_results]
        else:
            raise Exception("Model trained without parameters grid, no configurations available!")


def get_classification_prediction_data(model: object, X_test: np.ndarray, y_test: np.ndarray, classes: list) -> list[dict]:
    """
    Generates classification prediction data for each class in a multi-class classification problem.
    This data includes precision-recall, ROC curve data, and DET curve data.

    Parameters
    ----------
    model : Classifier object
        The trained classification model.
    X_test : np.ndarray
        The features for the test set.
    y_test : np.ndarray
        The true labels for the test set. 
    classes : list
        List of classes in the classification problem.

    Returns
    -------
    data_per_class : list of dict
        For each class, a dictionary containing precision, recall, FPR, TPR, ROC AUC, FNR and FAR values.
    """
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=classes)

    # Compute the prediction probabilities
    y_scores = model.predict_proba(X_test)

    # Prepare data for each class
    data_per_class = []
    for i in range(len(classes)):
        # Compute the precision-recall
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        # Compute the DET curve
        fnr, far, _ = det_curve(y_test_bin[:, i], y_scores[:, i])

        data_per_class.append({
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'fnr': fnr,
            'far': far,
        })
    
    return data_per_class





