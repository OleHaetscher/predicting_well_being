import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

pd.options.mode.chained_assignment = None  # default='warn'


class Imputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer class that can handle different types of imputation (linear or non-linear).
    Can be used in an sklearn pipeline.
    """

    def __init__(self, model, fix_rs, num_imputations, max_iter):
        self.model = model
        self.fix_rs = fix_rs
        self.max_iter = max_iter
        self.num_imputations = num_imputations  # total num imputations

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
        """
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,  # Different seed for each imputation
            sample_posterior=True
        )
        imputer.fit(df)
        df_imputed = imputer.transform(df)
        return df_imputed

    def apply_nonlinear_imputations(self, df: pd.DataFrame, num_imputation: int) -> pd.DataFrame:
        """
        Applies recursive partitioning (using CART) to impute missing values in the dataframe.
        It handles both continuous and binary variables.
        See "Recursive partitioning for missing data imputation in the presence of interaction effects"
        from Doove et al (2014) for details

        """
        np.random.seed(self.fix_rs + num_imputation)
        df_imputed = df.copy()

        # Columns with missing values ordered by the number of missing values
        columns_with_na = df.isna().sum().sort_values(ascending=True)
        columns_with_na = columns_with_na[columns_with_na > 0].index  # Only columns with missing values

        # Initial imputation by random sampling from observed values
        for col in columns_with_na:
            obs_values = df_imputed[col].dropna()
            missing_indices = df_imputed.index[df_imputed[col].isna()]
            df_imputed.loc[missing_indices, col] = np.random.choice(
                obs_values, size=len(missing_indices)
            )

        # Iterative process to refine imputations
        for iteration in range(self.max_iter):
            for col in columns_with_na:
                # Separate observed and missing data for the current column
                observed_mask = df[col].notna()
                missing_mask = df[col].isna()

                Y_obs = df_imputed.loc[observed_mask, col]
                X_obs = df_imputed.loc[observed_mask, df.columns != col]
                X_mis = df_imputed.loc[missing_mask, df.columns != col]

                if len(X_obs) == 0 or len(X_mis) == 0:
                    continue

                # Check if the column is continuous or binary/categorical
                if pd.api.types.is_numeric_dtype(Y_obs) and len(Y_obs.unique()) > 2:
                    # Continuous variable: Use DecisionTreeRegressor
                    tree_model = DecisionTreeRegressor(random_state=self.fix_rs)
                else:
                    # Binary or categorical variable: Use DecisionTreeClassifier
                    tree_model = DecisionTreeClassifier(random_state=self.fix_rs)

                # Fit the model on observed data
                tree_model.fit(X_obs, Y_obs)

                # Predict leaves for missing data
                leaves_for_missing = tree_model.apply(X_mis)
                leaves_for_observed = tree_model.apply(X_obs)

                # Get the indices of missing rows
                missing_indices = df_imputed.index[missing_mask].to_numpy()

                # Assign missing values by randomly sampling from observed values within the same leaf
                for leaf in np.unique(leaves_for_missing):
                    # Get the donors (observed values) from the same leaf
                    donors = Y_obs[leaves_for_observed == leaf]

                    # Get the positions of the missing values that fall in the same leaf
                    indices_in_leaf = np.where(leaves_for_missing == leaf)[0]

                    # Get the DataFrame indices of the missing values in this leaf
                    leaf_missing_indices = missing_indices[indices_in_leaf]

                    # For each missing value, randomly choose a donor from the same leaf
                    if len(donors) > 0:
                        imputed_values = np.random.choice(donors, size=len(indices_in_leaf))
                        df_imputed.loc[leaf_missing_indices, col] = imputed_values

        return df_imputed


