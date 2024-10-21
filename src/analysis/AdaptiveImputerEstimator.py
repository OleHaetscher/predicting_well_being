import numpy as np
from sklearn.base import BaseEstimator


class AdaptiveImputerEstimator(BaseEstimator):

    def __init__(self, regressor, classifier, categorical_idx,
                 feat_idx=None, neighbor_feat_idx=None,
                 **params
                 ):
        self.regressor = regressor
        self.classifier = classifier
        self.categorical_idx = categorical_idx
        self.feat_idx = feat_idx
        self.neighbor_feat_idx = neighbor_feat_idx
        self.model = None

    def is_categorical(self, X, feat_idx):
        return feat_idx in self.categorical_idx

    def fit(self, X, y):
        X = np.array(X)
        if self.is_categorical(X, self.feat_idx):
            self.model = self.classifier
        else:
            self.model = self.regressor

        self.model.fit(X, y)
        return self

    def predict(self, X, return_std):
        if return_std:
            return self.model.predict(X, return_std=return_std)
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


