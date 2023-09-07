import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesGridSearch:
    def __init__(self, alphas=None):
        if alphas is None:
            alphas = [1.0]  # Default alpha value

        self.grid_search = GridSearchCV(estimator=MultinomialNB(),
                                        param_grid={'alpha': alphas},
                                        scoring='accuracy',
                                        cv=5,
                                        verbose=0)
        self.classifier = None

    def fit(self, X, y):
        self.grid_search.fit(X, y)
        self.classifier = self.grid_search.best_estimator_

    def predict(self, X):
        if self.classifier is not None:
            return self.classifier.predict(X)
        else:
            raise ValueError("The classifier has not been trained yet. Please call 'fit' first.")

    def best_parameters(self):
        return self.grid_search.best_params_