import xgboost as xgb
from sklearn.metrics import accuracy_score

class XGBoost:
    def __init__(self, params=None):
        if params is None:
            params = {
                'objective': 'multi:softmax',
                'num_class': 2,  # Number of classes
                'eta': 0.3,       # Learning rate
                'max_depth': 6,   # Maximum depth of the tree
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 1,
                'silent': 1,
            }

        self.params = params
        self.num_round = 10  # Number of boosting rounds
        self.model = None

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, self.num_round)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        return y_pred.astype(int)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
