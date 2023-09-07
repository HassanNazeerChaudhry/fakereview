from sklearn import svm
from sklearn.model_selection import GridSearchCV

class SVMGridSearch:
    def __init__(self, kernel='rbf', C_range=None, gamma_range=None):
        self.kernel = kernel
        self.C_range = C_range if C_range else [1.0]  # Default C value
        self.gamma_range = gamma_range if gamma_range else ['scale']  # Default gamma value

        self.param_grid = {
            'C': self.C_range,
            'gamma': self.gamma_range
        }

        self.grid_search = GridSearchCV(estimator=svm.SVC(kernel=self.kernel),
                                        param_grid=self.param_grid,
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
