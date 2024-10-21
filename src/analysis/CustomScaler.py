import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    CustomScaler scales only the continuous columns and ignores the binary columns.
    It automatically determines which columns are continuous based on the data provided during fitting.
    This class is compatible with scikit-learn's Pipeline and metadata routing.

    Attributes:
        scaler: StandardScaler object from sklearn, used to scale continuous features.
        continuous_cols: List of continuous feature names determined during fitting.
        binary_cols: List of binary feature names determined during fitting.
    """

    def __init__(self):
        """
        Initializes the CustomScaler instance.
        """
        self.scaler = StandardScaler()
        self.continuous_cols = None
        self.binary_cols = None
        self.other_cols = None

    def fit(self, X, y=None):
        """
        Fits the scaler to the continuous columns in X.

        Args:
            X (pd.DataFrame): The input features to fit.
            y: Ignored. Present for compatibility.

        Returns:
            self: The fitted CustomScaler instance.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Determine continuous and binary columns
        self._separate_binary_continuous_cols(X)

        # Fit the scaler to the continuous columns
        if self.continuous_cols:
            self.scaler.fit(X[self.continuous_cols])

        return self

    def transform(self, X, y=None):
        """
        Transforms X by scaling the continuous columns and leaving binary columns unchanged.

        Args:
            X (pd.DataFrame): The input features to transform.
            y: Ignored. Present for compatibility.

        Returns:
            X_processed (pd.DataFrame): The transformed features, with columns in the same order as X.
        """
        # Ensure X is a DataFrame
        X_cont = X[self.continuous_cols]
        try: # a bit messy, but ok
            X_other = X[self.other_cols]
        except KeyError:
            X_other = pd.DataFrame()

        if not isinstance(X_cont, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Check if fit has been called
        if self.continuous_cols is None or self.binary_cols is None:
            raise RuntimeError("CustomScaler has not been fitted yet.")

        # Scale continuous columns
        if self.continuous_cols:
            X_scaled = self.scaler.transform(X_cont)
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.continuous_cols, index=X_cont.index)
        else:
            X_scaled_df = pd.DataFrame(index=X_cont.index)

        # Get binary columns
        X_binary = X[self.binary_cols] if self.binary_cols else pd.DataFrame(index=X_cont.index)

        # Combine scaled and binary dataframes
        X_processed_df = pd.concat([X_scaled_df, X_binary, X_other], axis=1)

        # Ensure the final dataframe preserves the original column order
        X_processed_df = X_processed_df[X.columns]

        return X_processed_df

    def inverse_transform(self, X, y=None):
        """
        Inverse-transforms the scaled features back to their original values.

        Args:
            X (pd.DataFrame): The transformed features to inverse-transform.
            y: Ignored. Present for compatibility.

        Returns:
            X_original_df (pd.DataFrame): The features in their original scale and order.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Check if fit has been called
        if self.continuous_cols is None or self.binary_cols is None:
            raise RuntimeError("CustomScaler has not been fitted yet.")

        # Inverse transform continuous columns
        if self.continuous_cols:
            X_continuous = X[self.continuous_cols]
            X_continuous_original = self.scaler.inverse_transform(X_continuous)
            X_continuous_original_df = pd.DataFrame(X_continuous_original, columns=self.continuous_cols, index=X.index)
        else:
            X_continuous_original_df = pd.DataFrame(index=X.index)

        # Get binary columns
        X_binary = X[self.binary_cols] if self.binary_cols else pd.DataFrame(index=X.index)

        # Combine the data
        X_original_df = pd.concat([X_continuous_original_df, X_binary], axis=1)

        # Ensure the columns are in the same order as original
        X_original_df = X_original_df[X.columns]

        return X_original_df

    def _separate_binary_continuous_cols(self, X):
        """
        Determines which features are continuous and which are binary based on the data in X.

        Args:
            X (pd.DataFrame): The input features to analyze.

        Sets:
            self.binary_cols: List of binary feature names.
            self.continuous_cols: List of continuous feature names.
        """
        data = X.copy()

        # Remove columns that start with "other_"
        data = data.drop([col for col in data.columns if col.startswith("other_")], axis=1)

        # Determine binary columns: columns containing only 0, 1, or NaN
        binary_cols = data.columns[(data.isin([0, 1]) | data.isna()).all()]
        continuous_cols = [col for col in data.columns if col not in binary_cols]  # Ordered as in X
        other_cols = [col for col in X.columns if col.startswith("other_")]

        self.binary_cols = binary_cols.tolist()
        self.continuous_cols = continuous_cols
        self.other_cols = other_cols
