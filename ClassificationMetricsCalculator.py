from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassificationMetricsCalculator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)

    def compute_all_metrics(self):
        accuracy = self.accuracy()
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1_score()
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
