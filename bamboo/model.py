import joblib
import os
from sklearn.calibration import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve, auc, det_curve
from sklearn.experimental import enable_halving_search_cv, enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

class Model:
    def __init__(self, model: object = None, params: dict = None, search_type: str = "GridSearch"):
        self.model = model
        self.params = params
        self.search = None
        self.search_type = search_type

    def fit(self, X, y):
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

    def predict(self, X):
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

    def get_fit_results(self):
        if self.search:
            return self.search.cv_results_
        else:
            raise Exception("Model trained without parameters grid, no fit results available!")

    def get_best_configuration(self):
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

    def get_configurations(self, sort_by: str = 'Score', n_results: int = None, ascending: bool = False):
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

def get_classification_prediction_data(model, X_test, y_test, classes):
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





