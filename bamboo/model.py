import joblib
import os
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

    def get_summary(self):
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