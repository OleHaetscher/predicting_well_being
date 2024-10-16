import pickle

import pandas as pd
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from joblib import Parallel, delayed


class Imputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer class that can handle different types of imputation (linear or non-linear).
    Can be used in an sklearn pipeline.
    """

    def __init__(
        self,
        model,
        fix_rs,
        num_imputations,
        max_iter,
        n_jobs_imputation_columns,
        conv_thresh,
        percentage_of_features,
        n_features_thresh,
        country_group_by,
        logger
    ):
        self.model = model
        self.fix_rs = fix_rs
        self.num_imputations = num_imputations
        self.max_iter = max_iter
        self.n_jobs_imputation_columns = n_jobs_imputation_columns
        self.conv_thresh = conv_thresh  # only for RFR
        self.percentage_of_features = percentage_of_features  # only for ENR
        self.n_features_thresh = n_features_thresh
        self.country_group_by = country_group_by
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

        # Identify country-level variables
        country_var_cols = [col for col in df.columns if col.startswith('mac_')]

        if country_var_cols:
            df = self.impute_country_level(df=df, country_var_cols=country_var_cols, num_imputation=num_imputation)

        # Drop "other" columns as they should not be used for individual level imputations, identify individual columns
        other_var_cols = pd.DataFrame({col: df.pop(col) for col in df.columns if col.startswith('other_')})
        individual_var_cols = [col for col in df.columns if not col.startswith('mac_')]

        # Proceed with individual-level variables (also use if condition, in mac, we do not have any individual columns)
        if individual_var_cols:
            df = self.impute_individual_level(df=df, individual_var_cols=individual_var_cols, num_imputation=num_imputation)

        # Created df, merge other columns back to df
        df_imputed = pd.DataFrame(df, columns=df.columns, index=df.index)
        df_imputed = pd.concat([df_imputed, other_var_cols], axis=1)
        # Remove the country column (I do not need it anymore)
        df_imputed = df_imputed.drop(self.country_group_by, axis=1)
        return df_imputed

    def impute_country_level(self, df, country_var_cols, num_imputation):
        """
        Imputes missing values in country-level variables, ensuring that participants from the same country
        receive the same imputed values.
        """
        # Create DataFrame with one row per country
        # TODO: Group by year and tuple, when merging back take the mean if years are two different values
        country_df = df.groupby(self.country_group_by)[country_var_cols].first()

        # Apply the appropriate imputation method
        if self.model == 'elasticnet':
            country_array_imputed = self.apply_linear_imputations(
                df=country_df,
                num_imputation=num_imputation,
            )
        elif self.model == 'randomforestregressor':
            country_array_imputed = self.apply_nonlinear_imputations(
                df=country_df,
                num_imputation=num_imputation,
                n_jobs=self.n_jobs_imputation_columns
            )
        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        # Merge back the imputed country-level data to the original DataFrame
        country_df_imputed = pd.DataFrame(country_array_imputed, columns=country_df.columns, index=country_df.index).reset_index()
        df = df.drop(columns=country_var_cols)
        df_merged = df.merge(country_df_imputed, on=self.country_group_by, how='left')
        df_merged.index = df.index

        return df_merged

    def impute_individual_level(self, df, individual_var_cols, num_imputation):
        """
        Imputes missing values on the individual level.
        """
        individual_df = df[individual_var_cols]

        if self.model == 'elasticnet':
            individual_array_imputed = self.apply_linear_imputations(
                df=individual_df,
                num_imputation=num_imputation
            )
        elif self.model == 'randomforestregressor':
            individual_array_imputed = self.apply_nonlinear_imputations(
                df=individual_df,
                num_imputation=num_imputation
            )
        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        # Merge
        individual_df_imputed = pd.DataFrame(individual_array_imputed, columns=individual_df.columns, index=individual_df.index)
        df = df.drop(columns=individual_var_cols)
        df = pd.concat([df, individual_df_imputed], axis=1, join="outer")

        return df

    def apply_linear_imputations(self, df: pd.DataFrame, num_imputation: int) -> pd.DataFrame:
        """
        Applies linear imputations using the IterativeImputer from sklearn.
        In analysis with many features, we reduce the number of features used for imputation.
        """

        n_features = int(len(df.columns) * self.percentage_of_features)
        if n_features < self.n_features_thresh:
            n_features = None  # Use all features

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

            for col, imputed_series in results:
                if imputed_series is not None:
                    missing_indices = missing_indices_dict[col]
                    new_values = imputed_series.values

                    # Update df_imputed
                    df_imputed.loc[missing_indices, col] = new_values
        gc.collect()

        return df_imputed
