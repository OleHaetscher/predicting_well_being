from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class PearsonFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=None, correlation_threshold=None, target_prefix="sens_"):
        """
        Initialize the PearsonFeatureSelector.

        Args:
            num_features (int, optional): The number of top features to select based on absolute correlation.
            correlation_threshold (float, optional): The minimum absolute correlation to consider a feature informative.
            target_prefix (str): The prefix used to identify features to be selected based on correlation (default is "sens_").
        """
        if num_features is None and correlation_threshold is None:
            raise ValueError("Either num_features or correlation_threshold must be specified.")

        self.num_features = num_features
        self.correlation_threshold = correlation_threshold
        self.target_prefix = target_prefix
        self.selected_features_ = None

    def fit(self, X, y):
        """
        Fit the selector by computing Pearson correlations for specified features and selecting the top ones.

        Args:
            X (pd.DataFrame): The feature dataset.
            y (pd.Series or pd.DataFrame): The target variable.

        Returns:
            self: Fitted selector with selected features.
        """
        # Identify features that match the target prefix
        target_features = [col for col in X.columns if col.startswith(self.target_prefix)]

        if not target_features:
            raise ValueError(f"No features found with the prefix '{self.target_prefix}'.")

        # Compute Pearson correlation between each target feature and the target variable
        correlations = X[target_features].apply(lambda col: col.corr(y))
        abs_correlations = correlations.abs()  # Get absolute values of the correlations

        # Select features based on the specified method
        if self.num_features is not None:
            # Select top `num_features` based on absolute correlation
            selected_features = abs_correlations.nlargest(self.num_features).index.tolist()
        elif self.correlation_threshold is not None:
            # Select features with absolute correlation above the threshold
            selected_features = abs_correlations[abs_correlations >= self.correlation_threshold].index.tolist()

        # Store selected target features along with non-target features
        self.selected_features_ = selected_features + [col for col in X.columns if not col.startswith(self.target_prefix)]

        return self

    def transform(self, X):
        """
        Transform the dataset by selecting the previously identified features.

        Args:
            X (pd.DataFrame): The feature dataset to transform.

        Returns:
            pd.DataFrame: Transformed dataset with only the selected features.
        """
        # Ensure the selector has been fitted
        if self.selected_features_ is None:
            raise RuntimeError("You must fit the selector before transforming data.")

        # Return only the selected features
        return X[self.selected_features_]

    def get_support(self, indices=False):
        """
        Get the mask or indices of selected features.

        Args:
            indices (bool): If True, return the indices of selected features.
                            If False, return a boolean mask of the selected features.

        Returns:
            np.ndarray: Indices or boolean mask of the selected features.
        """
        if self.selected_features_ is None:
            raise RuntimeError("You must fit the selector before getting support.")

        feature_indices = [i for i, col in enumerate(self.selected_features_)]
        if indices:
            return feature_indices
        else:
            mask = [False] * len(self.selected_features_)
            for idx in feature_indices:
                mask[idx] = True
            return mask