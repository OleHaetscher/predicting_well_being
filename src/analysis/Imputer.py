import gc
import os
import pickle
import threading

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from src.utils.Logger import Logger

pd.options.mode.chained_assignment = None  # default='warn'


class Imputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer class that can handle different types of imputation (linear or non-linear).
    Can be used in an sklearn pipeline.
    """

    def __init__(self, model, fix_rs, num_imputations, max_iter, n_jobs_imputation_columns, conv_thresh, percentage_of_features, logger):
        self.model = model
        self.fix_rs = fix_rs
        self.num_imputations = num_imputations
        self.max_iter = max_iter
        self.n_jobs_imputation_columns = n_jobs_imputation_columns
        self.conv_thresh = conv_thresh # only for RFR
        self.percentage_of_features = percentage_of_features  # only for ENR
        self.logger = logger

    def fit(self, X, y=None):
        """
        In this case, the fit method is not really necessary, but it is required to conform to sklearn's API.
        """
        return self

    def transform(self, X, y=None, num_imputation=None):
        """
        Applies the appropriate imputation method based on the model type.
        """
        df = X.copy()
        if self.model == 'elasticnet':
            X_imputed = self.apply_linear_imputations(df=df, num_imputation=num_imputation)
        elif self.model == 'randomforestregressor':
            X_imputed = self.apply_nonlinear_imputations(df=df, num_imputation=num_imputation)
        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")
        return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    def apply_linear_imputations(self, df: pd.DataFrame, num_imputation: int) -> pd.DataFrame:
        """
        Applies linear imputations using the IterativeImputer from sklearn.
        To reduce computational complexity (which can be crazy in the analysis including all datasets and sensing features), we
            - reduce the number of features used for imputation dynamically
            - skip complete features
        """
        n_features = len(df.columns) // (1 / self.percentage_of_features)

        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,  # Different seed for each imputation
            sample_posterior=True,
            skip_complete=False,
            n_nearest_features=n_features,
            verbose=0,
        )

        imputer.fit(df)
        df_imputed = imputer.transform(df)
        return df_imputed

    def apply_nonlinear_imputations(self, df: pd.DataFrame, num_imputation: int, n_jobs: int = -1) -> pd.DataFrame:
        """
        Applies recursive partitioning (using CART) to impute missing values in the dataframe.
        It handles both continuous and binary variables.
        See "Recursive partitioning for missing data imputation in the presence of interaction effects"
        from Doove et al (2014) for details.
        """
        np.random.seed(self.fix_rs + num_imputation)
        df_imputed = df.copy()

        # Convert data types to save memory (e.g., float64 to float32)
        for col in df_imputed.select_dtypes(include=['float64']).columns:
            df_imputed[col] = df_imputed[col].astype('float32')
        for col in df_imputed.select_dtypes(include=['int64']).columns:
            df_imputed[col] = df_imputed[col].astype('int32')

        # Columns with missing values ordered by the number of missing values
        columns_with_na = df.isna().sum().sort_values(ascending=True)
        columns_with_na = columns_with_na[columns_with_na > 0].index  # Only columns with missing values

        # Initial imputation by random sampling from observed values
        missing_indices_dict = {}  # To store indices of missing values per column
        prev_imputed_values = {}  # To store previous imputed values for convergence check

        for col in columns_with_na:
            obs_values = df_imputed[col].dropna().values
            missing_indices = df.index[df[col].isna()]  # Indices where original data is missing
            df_imputed.loc[missing_indices, col] = np.random.choice(
                obs_values, size=len(missing_indices)
            )
            # Store indices and initial imputed values
            missing_indices_dict[col] = missing_indices
            prev_imputed_values[col] = df_imputed.loc[missing_indices, col].copy()

        # Function to process one column
        def process_column(col):
            # Remove, just for testing parallelism

            # Read from df_imputed at the start of the iteration
            observed_mask = df[col].notna().values
            missing_mask = df[col].isna().values

            y_obs = df_imputed.loc[observed_mask, col].values
            X_obs = df_imputed.loc[observed_mask, df.columns != col].values
            X_mis = df_imputed.loc[missing_mask, df.columns != col].values

            if X_obs.shape[0] == 0 or X_mis.shape[0] == 0:
                return col, None  # No update

            # Check if the column is continuous or binary/categorical
            if pd.api.types.is_numeric_dtype(y_obs) and len(np.unique(y_obs)) > 2:
                tree_model = DecisionTreeRegressor(
                    random_state=self.fix_rs,
                    max_depth=10,  # Limit depth to speed up
                )
            else:
                tree_model = DecisionTreeClassifier(
                    random_state=self.fix_rs,
                    max_depth=10,  # Limit depth to speed up
                )

                # Ensure y_obs is correctly encoded
                unique_values = np.unique(y_obs)
                if len(unique_values) > 2 or not np.array_equal(unique_values, [0, 1]):
                    # Map to 0 and 1
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    y_obs = np.array([value_map[val] for val in y_obs])

            # Fit the model on observed data
            tree_model.fit(X_obs, y_obs)

            # Predict leaves for missing data
            leaves_for_missing = tree_model.apply(X_mis)
            leaves_for_observed = tree_model.apply(X_obs)

            # Get the indices of missing rows
            missing_indices = df_imputed.index[missing_mask].to_numpy()

            # Assign missing values by randomly sampling from observed values within the same leaf
            imputed_values = np.empty(len(missing_indices), dtype=df_imputed[col].dtype)
            leaves_unique = np.unique(leaves_for_missing)

            for leaf in leaves_unique:
                # Get the donors (observed values) from the same leaf
                donor_mask = leaves_for_observed == leaf
                donors = y_obs[donor_mask]

                # Get the positions of the missing values that fall in the same leaf
                indices_in_leaf = np.where(leaves_for_missing == leaf)[0]

                # For each missing value, randomly choose a donor from the same leaf
                if donors.size > 0:
                    imputed_values[indices_in_leaf] = np.random.choice(donors, size=len(indices_in_leaf))
                else:
                    # If no donors are available, impute with the overall mean or mode
                    if np.issubdtype(df_imputed[col].dtype, np.number):
                        imputed_values[indices_in_leaf] = np.mean(y_obs)
                    else:
                        imputed_values[indices_in_leaf] = pd.Series(y_obs).mode()[0]

            # Update the DataFrame with imputed values
            imputed_series = pd.Series(imputed_values, index=missing_indices)

            # Clean up to save memory
            del X_obs, X_mis, y_obs, tree_model, leaves_for_missing, leaves_for_observed
            gc.collect()

            return col, imputed_series

        # Iterative process to refine imputations with convergence check
        for iteration in range(self.max_iter):
            # For each column, process in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_column)(col) for col in columns_with_na
            )

            # Collect the updates and check for convergence
            converged = True  # Assume converged unless proven otherwise
            for col, imputed_series in results:
                if imputed_series is not None:
                    missing_indices = missing_indices_dict[col]
                    prev_values = prev_imputed_values[col].values
                    new_values = imputed_series.values

                    # Update df_imputed
                    df_imputed.loc[missing_indices, col] = new_values

                    # Compare the new imputed values to previous
                    if pd.api.types.is_numeric_dtype(df_imputed[col]):
                        # For numeric data
                        diff = np.abs(new_values - prev_values)
                        max_diff = np.nanmax(diff)
                        if max_diff > self.conv_thresh:
                            converged = False
                    else:
                        # For categorical or object data
                        changed = prev_values != new_values
                        num_changed = np.sum(changed)
                        if num_changed > 0:
                            converged = False

                    # Update previous imputed values
                    prev_imputed_values[col] = imputed_series.copy()

            if converged:
                self.logger.log(f"Converged at iteration {iteration}")
                break

        # Optionally, clean up memory after each iteration
        gc.collect()

        self.logger.log(f"      Imputation {num_imputation} not converged based on convergence threshold "
                        f"{self.conv_thresh} before max_iter {self.max_iter}")

        ### Just for testing
        with open("test_imputed_dataset", 'wb') as f:
            pickle.dump(df_imputed, f)

        return df_imputed


