import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    This class scales only the continuous columns and ignore the binary columns. This custom class
    works with metadata_routing, which does not work with using the ColumnTransformer.

    Attributes:
         cols_to_scale: Columns indicating the continuous features that should be scaled.
         scaler: Standardscaler object from sklearn, calculates z scores [(x - M) / SD]
    """

    def __init__(self, cols_to_scale):
        """
        Constructor method of the CustomScaler class.

        Args:
            cols_to_scale: Columns indicating the continuous features that should be scaled.
        """
        self.cols_to_scale = cols_to_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        This method fits the scaler to the selected columns.

        Args:
            X: df, features to be scaled.

        Returns:
            self: the CustomScaler object itself.
        """
        self.scaler.fit(X[self.cols_to_scale])
        return self

    def transform(self, X, y=None):
        """
        This method transforms the selected columns using the scaler object, and returns a numpy array
        with the columns in the same order as in the original dataframe.

        Args:
            X: df, features to be scaled.

        Returns:
            X_processed: ndarray, containing the scaled features in the same order as original.
        """
        X_scaled = self.scaler.transform(X[self.cols_to_scale])  # Scaled continuous columns
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.cols_to_scale, index=X.index)  # Recreate a DataFrame
        X_unscaled = X.drop(self.cols_to_scale, axis=1)  # Unscaled binary columns

        # Combine scaled and unscaled dataframes
        X_processed_df = pd.concat([X_scaled_df, X_unscaled], axis=1)

        # Ensure the final dataframe preserves the original column order
        X_processed_df = X_processed_df[X.columns]

        return X_processed_df

    def inverse_transform(self, X, y=None):
        """
        This method inverse-transforms the scaled features back to their original values.

        Args:
            X: DataFrame, containing the scaled features.

        Returns:
            X_original_df: DataFrame, containing the features in their original scale and order.
        """
        # Separate scaled and unscaled data
        X_scaled = X[self.cols_to_scale]
        X_unscaled = X.drop(self.cols_to_scale, axis=1)

        # Apply inverse scaling to the scaled columns
        X_scaled_original = self.scaler.inverse_transform(X_scaled)
        X_scaled_original_df = pd.DataFrame(X_scaled_original, columns=self.cols_to_scale, index=X.index)

        # Combine the data
        X_original_df = pd.concat([X_scaled_original_df, X_unscaled], axis=1)

        # Ensure the columns are in the same order as original
        X_original_df = X_original_df[X.columns]

        return X_original_df
