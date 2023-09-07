from sklearn import svm

class LinearSVM:
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.classifier = svm.SVC(kernel='linear', C=self.C)

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)
