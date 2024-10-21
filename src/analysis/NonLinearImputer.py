import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import gc

class NonLinearImputer:
    def __init__(self, max_iter=10, random_state=None, tree_max_depth=None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.models_ = {}        # Stores models for each column
        self.fallback_values_ = {}  # Fallback
        self.columns_ = None     # Stores the column names
        self.categorical_mappings_ = {}  # To store mappings for categorical variables
        self.tree_max_depth = tree_max_depth

    def fit(self, X, y=None):
        """
        Fit the imputer on X.
        """
        np.random.seed(self.random_state)
        df_imputed = X.copy()
        self.columns_ = df_imputed.columns

        # Columns with missing values, ordered by increasing number of missing values
        na_counts = df_imputed.isna().sum()
        columns_with_na = na_counts[na_counts > 0].sort_values().index.tolist()

        missing_indices_dict = {}

        # Initial imputation by random sampling from observed values
        for col in columns_with_na:
            obs_values = df_imputed[col].dropna().values
            missing_indices = df_imputed.index[df_imputed[col].isna()]
            if len(obs_values) > 0:
                df_imputed.loc[missing_indices, col] = np.random.choice(
                    obs_values, size=len(missing_indices)
                )
            else:
                # If no observed values, impute with zeros or some default value
                df_imputed.loc[missing_indices, col] = 0  # or np.nan

            missing_indices_dict[col] = missing_indices

        # Handle columns with no missing values in training (for potential missing in test)
        for col in df_imputed.columns:
            # Get unique non-NaN values in the column
            unique_values = df_imputed[col].dropna().unique()

            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):  # Binary column (0/1)
                # Store the mode for binary columns (0/1)
                self.fallback_values_[col] = df_imputed[col].mode()[0]
            else:
                # Store the mean for continuous columns
                self.fallback_values_[col] = df_imputed[col].mean()

            print(f"Stored fallback for column '{col}': {self.fallback_values_[col]}")

        # Iterative imputation process
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration+1}/{self.max_iter}")
            for col in columns_with_na:
                observed_mask = df_imputed[col].notna()
                y_obs = df_imputed.loc[observed_mask, col]
                X_obs = df_imputed.loc[observed_mask, df_imputed.columns != col]

                if X_obs.shape[0] == 0:
                    continue  # No observed data to fit the model

                # Determine if the column is continuous or categorical
                if pd.api.types.is_numeric_dtype(y_obs) and len(np.unique(y_obs)) > 2:
                    tree_model = DecisionTreeRegressor(
                        random_state=self.random_state,
                        max_depth=self.tree_max_depth,
                    )
                else:
                    tree_model = DecisionTreeClassifier(
                        random_state=self.random_state,
                        max_depth=self.tree_max_depth
                    )
                    # Encode categorical variables
                    y_obs, uniques = pd.factorize(y_obs)
                    self.categorical_mappings_[col] = uniques

                # Fit the model
                tree_model.fit(X_obs, y_obs)

                # Use the stored missing indices from the initial imputation
                missing_indices = missing_indices_dict[col]
                X_mis = df_imputed.loc[missing_indices, df_imputed.columns != col]

                # Impute values
                self._impute_values(df_imputed, col, tree_model, y_obs, X_obs, X_mis, missing_indices)

                # Update the model for this column
                self.models_[col] = tree_model

        # Store the final imputed DataFrame
        self._df_imputed_ = df_imputed

        return self

    def _impute_values(self, df_imputed, col, model, y_obs, X_obs, X_mis, missing_indices):
        # Apply the model to get leaves for observed and missing data
        leaves_for_missing = model.apply(X_mis)
        leaves_for_observed = model.apply(X_obs)

        y_obs_array = np.array(y_obs)

        imputed_values = np.empty(len(X_mis), dtype=df_imputed[col].dtype)
        leaves_unique = np.unique(leaves_for_missing)

        for leaf in leaves_unique:
            donor_mask = leaves_for_observed == leaf
            donors = y_obs_array[donor_mask]
            indices_in_leaf = np.where(leaves_for_missing == leaf)[0]

            if donors.size > 0:
                imputed_values_leaf = np.random.choice(donors, size=len(indices_in_leaf), replace=True)
                imputed_values[indices_in_leaf] = imputed_values_leaf
            else:
                # If no donors are available, impute with the overall mean or mode
                if np.issubdtype(df_imputed[col].dtype, np.number):
                    imputed_values[indices_in_leaf] = np.mean(y_obs_array)
                else:
                    imputed_values[indices_in_leaf] = pd.Series(y_obs_array).mode()[0]

        # For categorical variables, map back to original categories
        if col in self.categorical_mappings_:
            uniques = self.categorical_mappings_[col]
            imputed_values = uniques[imputed_values.astype(int)]

        df_imputed.loc[missing_indices, col] = imputed_values

    def transform(self, X):
        """
        Impute missing values in X using the fitted models.
        """
        df = X.copy()
        if not all(col in df.columns for col in self.columns_):
            raise ValueError("Input data must contain the same columns as during fitting.")

        # Columns with missing values
        columns_with_na = df.columns[df.isna().any()].tolist()

        # Impute missing values
        for col in columns_with_na:
            missing_mask = df[col].isna()

            if col not in self.models_:  # Use Fallback if col had no NaNs in the train set, but has NaNs in the test set
                if col in self.fallback_values_:
                    # Impute missing values with the stored fallback value (mean/mode)
                    df.loc[missing_mask, col] = self.fallback_values_[col]
                continue  # Skip to the next column if fallback value is used

            model = self.models_[col]

            # Fill missing values with a random sample from observed values
            obs_values = df[col].dropna().values
            if len(obs_values) > 0:
                df.loc[missing_mask, col] = np.random.choice(
                    obs_values, size=missing_mask.sum()
                )

            X_mis = df.loc[missing_mask, df.columns != col]

            # Use observed data from training set
            X_obs = self._df_imputed_.loc[self._df_imputed_[col].notna(), self._df_imputed_.columns != col]
            y_obs = self._df_imputed_.loc[self._df_imputed_[col].notna(), col]

            # For categorical variables, apply the same encoding as during fitting
            if col in self.categorical_mappings_:
                uniques = self.categorical_mappings_[col]
                y_obs = pd.Categorical(y_obs, categories=uniques).codes
            else:
                y_obs = np.array(y_obs)

            self._impute_values(df, col, model, y_obs, X_obs, X_mis, missing_mask)

        # Check if there are any NaN values left and fill with fallback if needed
        for col in df.columns:
            if df[col].isna().any():
                # Fill NaNs with the fallback value for this column
                if col in self.fallback_values_:
                    fallback_value = self.fallback_values_[col]
                    print(f"Filling NaNs in column '{col}' with fallback value: {fallback_value}")
                    df[col].fillna(fallback_value, inplace=True)
        return df
