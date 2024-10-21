import numpy as np
from sklearn.linear_model import LogisticRegression


class SafeLogisticRegression:
    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)
        self.single_class_ = None

    def fit(self, X, y):
        # Check if the target variable y has more than one unique class
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            # If only one class is present, store the class and skip fitting the model
            self.single_class_ = unique_classes[0]
        else:
            # Otherwise, fit the LogisticRegression model normally
            self.model.fit(X, y)

    def predict(self, X):
        if self.single_class_ is not None:
            # If only one class was present during fit, return that class for all predictions
            return np.full(X.shape[0], self.single_class_)
        else:
            # Otherwise, use the fitted LogisticRegression model to predict
            return self.model.predict(X)

    def predict_proba(self, X):
        if self.single_class_ is not None:
            # If only one class was present during fit, return probabilities of 1 for that class
            proba = np.zeros((X.shape[0], 1))
            return np.hstack([proba, np.ones((X.shape[0], 1))]) if self.single_class_ == 1 else np.hstack([np.ones((X.shape[0], 1)), proba])
        else:
            # Otherwise, use the fitted LogisticRegression model to predict probabilities
            return self.model.predict_proba(X)