import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}  # P(class)
        self.feature_probs = {}  # P(feature | class)

    def fit(self, X, y):
        # Calculate class probabilities P(class)
        classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        for c, count in zip(classes, class_counts):
            self.class_probs[c] = count / total_samples

        # Calculate feature probabilities P(feature | class)
        num_features = X.shape[1]

        for c in classes:
            self.feature_probs[c] = {}
            class_samples = X[y == c]

            for feature_index in range(num_features):
                feature_values = class_samples[:, feature_index]
                total_feature_count = len(feature_values)

                for feature_value, count in zip(*np.unique(feature_values, return_counts=True)):
                    self.feature_probs[c][f'F{feature_index}_{feature_value}'] = count / total_feature_count

    def predict(self, X):
        predictions = []

        for sample in X:
            max_prob = -1
            predicted_class = None

            for c, class_prob in self.class_probs.items():
                feature_probs = [self.feature_probs[c].get(f'F{i}_{value}', 1e-10) for i, value in enumerate(sample)]
                prob = np.log(class_prob) + np.sum(np.log(feature_probs))

                if prob > max_prob:
                    max_prob = prob
                    predicted_class = c

            predictions.append(predicted_class)

        return predictions