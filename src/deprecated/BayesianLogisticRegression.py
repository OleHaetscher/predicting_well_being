import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid function
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array


class BayesianRidgeLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=100, tol=1e-4):
        self.alpha = alpha  # Prior precision
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if len(self.classes_) == 1:
            # Store the single class and return self
            self.single_class_ = self.classes_[0]
            self.coef_ = None
            self.covariance_ = None
            return self

        # If more than one class is present, proceed with binary classification
        if len(self.classes_) > 2:
            raise ValueError("This implementation supports binary classification only.")

        # Convert y to binary labels
        y_binary = np.where(y == self.classes_[1], 1, 0)

        # Add intercept term if necessary
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.n_features_in_ = X.shape[1]

        # Define the negative log-posterior
        def objective(w):
            z = X.dot(w)
            y_prob = expit(z)
            nll = -np.sum(y_binary * np.log(y_prob + 1e-9) + (1 - y_binary) * np.log(1 - y_prob + 1e-9))
            # Regularization term (exclude intercept if fit_intercept)
            if self.fit_intercept:
                w_reg = w[1:]
            else:
                w_reg = w
            reg = 0.5 * self.alpha * np.dot(w_reg, w_reg)
            return nll + reg

        # Gradient of the negative log-posterior
        def grad(w):
            z = X.dot(w)
            y_prob = expit(z)
            error = y_prob - y_binary
            gradient = X.T.dot(error)
            if self.fit_intercept:
                gradient[1:] += self.alpha * w[1:]
            else:
                gradient += self.alpha * w
            return gradient

        # Initial guess
        w0 = np.zeros(self.n_features_in_)

        # Optimize the negative log-posterior
        res = minimize(
            objective,
            w0,
            jac=grad,
            method='BFGS',
            tol=self.tol,
            options={'maxiter': self.max_iter}
        )

        # Store the coefficients
        self.coef_ = res.x

        # Compute the Hessian at the MAP estimate
        z = X.dot(self.coef_)
        y_prob = expit(z)
        W = y_prob * (1 - y_prob)  # Diagonal of the weight matrix
        X_weighted = X * W[:, np.newaxis]
        H = X.T.dot(X_weighted)

        # Add prior precision (regularization)
        if self.fit_intercept:
            H[1:, 1:] += self.alpha * np.eye(self.n_features_in_ - 1)
        else:
            H += self.alpha * np.eye(self.n_features_in_)

        # Compute the covariance matrix (inverse Hessian)
        self.covariance_ = np.linalg.inv(H)

        return self

    def predict_proba(self, X, return_std=False):
        # Handle the case when only one class is present
        if hasattr(self, 'single_class_'):
            prob = np.ones(X.shape[0]) if self.single_class_ == self.classes_[1] else np.zeros(X.shape[0])
            if return_std:
                return prob, np.zeros(X.shape[0])
            return prob
        # Check if fit has been called
        X = check_array(X)

        # Add intercept term if necessary
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute the mean predictions
        mu = X.dot(self.coef_)

        # Compute the variance for each sample
        sigma2 = np.sum(X.dot(self.covariance_) * X, axis=1)

        # Approximate predictive probabilities using probit approximation
        kappa = np.sqrt(1 + (np.pi * sigma2) / 8.0)
        prob = expit(mu / kappa)

        if return_std:
            # Compute standard deviation of the probabilities
            std = np.sqrt(prob * (1 - prob))

            return prob, std

        return prob

    def predict(self, X, return_std=False):
        # Handle the case when only one class is present
        if hasattr(self, 'single_class_'):
            if return_std:
                return np.full(X.shape[0], self.single_class_), np.zeros(X.shape[0])
            return np.full(X.shape[0], self.single_class_)
        # Predict class labels for samples in X
        prob = self.predict_proba(X, return_std=return_std)
        if return_std:
            prob, std = prob
            indices = (prob >= 0.5).astype(int)
            return self.classes_[indices], std
        else:
            indices = (prob >= 0.5).astype(int)
            return self.classes_[indices]