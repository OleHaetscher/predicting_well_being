import random

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.linear_model import Ridge

from src.analysis.AdaptiveImputerEstimator import AdaptiveImputerEstimator
from src.analysis.CustomIterativeImputer import CustomIterativeImputer
from src.analysis.CustomScaler import CustomScaler
from src.analysis.NonLinearImputer import NonLinearImputer
from src.analysis.SafeLogisticRegression import SafeLogisticRegression


class Imputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer class that can handle different types of imputation (linear or non-linear).
    Can be used in an sklearn pipeline.
    """

    def __init__(
        self,
        logger,
        model,
        fix_rs,
        num_imputations,
        max_iter,
        conv_thresh,
        tree_max_depth,
        percentage_of_features,
        n_features_thresh,
        sample_posterior,
        pmm_k,
        country_group_by,
        years_col,
    ):
        self.logger = logger
        self.model = model
        self.fix_rs = fix_rs
        self.num_imputations = num_imputations
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh  # only for RFR
        self.tree_max_depth = tree_max_depth  # only for RFR
        self.percentage_of_features = percentage_of_features  # only for ENR
        self.n_features_thresh = n_features_thresh   # only for ENR
        self.sample_posterior = sample_posterior
        self.pmm_k = pmm_k
        self.country_group_by = country_group_by
        self.years_col = years_col
        self.logger = logger
        self.country_imputer = None
        self.individual_imputer = None
        self.fitted_country_scaler = None
        self.fitted_individual_scaler = None

    def fit(self, X, y=None, num_imputation=None):
        """
        Fit the imputer for both country and individual level variables.

        Args:
            X: The input data to fit the imputation models.
            y: Ignored, present for compatibility.
            num_imputation: The number of imputations for missing data.

        Returns:
            self: The fitted imputer object.
        """
        df = X.copy()

        # Identify country-level variables (e.g., mac_ prefix)
        country_var_cols = [col for col in df.columns if col.startswith('mac_')]

        # Fit country-level imputer if needed
        if country_var_cols:
            self.logger.log(f"        Imputing country-level variables")
            # This will be either the linear or nonlinear imputer, depending on the analysis
            self.country_imputer = self._fit_country_level_imputer(
                df=df,
                country_var_cols=country_var_cols,
                num_imputation=num_imputation
            )

        individual_var_cols = [col for col in df.columns if not col.startswith('mac_') and not col.startswith('other_')]

        # Fit individual-level imputer if there are individual-level columns
        if individual_var_cols:
            self.logger.log(f"        Imputing individual-level variables")
            self.individual_imputer = self._fit_individual_level_imputer(
                df=df,
                individual_var_cols=individual_var_cols,
                num_imputation=num_imputation
            )

        return self

    def transform(self, X, y=None, num_imputation=None):
        """
        Applies the appropriate imputation method based on the model type.
        """
        df = X.copy()

        # Identify country-level variables (e.g., mac_ prefix)
        country_var_cols = [col for col in df.columns if col.startswith('mac_')]

        # Apply country-level imputation if imputer is fitted
        if country_var_cols and self.country_imputer:
            df = self._apply_country_level_imputation(
                df=df,
                country_var_cols=country_var_cols,
                num_imputation=num_imputation
            )

        # Drop "other" columns as they should not be used for individual-level imputations, identify individual columns
        other_var_cols = pd.DataFrame({col: df.pop(col) for col in df.columns if col.startswith('other_')})
        individual_var_cols = [col for col in df.columns if not col.startswith('mac_')]

        # Apply individual-level imputation if imputer is fitted
        if individual_var_cols and self.individual_imputer:
            df = self._apply_individual_level_imputation(
                df=df,
                individual_var_cols=individual_var_cols,
                num_imputation=num_imputation
            )

        # Combine back the imputed and non-imputed columns (like 'other_')
        df_imputed = pd.DataFrame(df, columns=df.columns, index=df.index)
        df_imputed = pd.concat([df_imputed, other_var_cols], axis=1)

        # Remove country group-by columns as they are not needed in the final dataset
        if self.country_group_by in df_imputed.columns:
            df_imputed = df_imputed.drop(self.country_group_by, axis=1)

        return df_imputed

    def _fit_country_level_imputer(self, df, country_var_cols, num_imputation):
        """
        Fit the country-level imputer.

        Args:
            df: DataFrame containing the country-level variables.
            country_var_cols: List of country-level variable columns.
            num_imputation: The number of imputations for missing data.

        Returns:
            Fitted imputer (e.g., statistics or models for imputation).
        """
        # Group df by country and year
        country_df, _, _ = self.prepare_country_df(df=df.copy(), country_var_cols=country_var_cols)

        # Fit Scaler, store as attribute (so that I can reuse the transform method in the apply method)
        scaler_country_vars = CustomScaler()
        self.fitted_country_scaler = scaler_country_vars.fit(country_df)
        country_df_scaled = self.fitted_country_scaler.transform(country_df)

        # Apply imputation on the country-level variables
        if self.model == 'elasticnet':
            country_imputer = self._fit_linear_imputer(
                df=country_df_scaled[country_var_cols],
                num_imputation=num_imputation,
            )
        elif self.model == 'randomforestregressor':
            country_imputer = self._fit_nonlinear_imputer(
                df=country_df_scaled[country_var_cols],
                num_imputation=num_imputation,
            )
        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        return country_imputer

    def _fit_individual_level_imputer(self, df, individual_var_cols, num_imputation):
        """
        Fit the individual imputer.

        Args:
            df: DataFrame containing the country-level variables.
            individual_var_cols: List of individual-level variable columns.
            num_imputation: The number of imputations for missing data.

        Returns:
            Fitted imputer (e.g., statistics or models for imputation).
        """
        individual_df = df[individual_var_cols]

        # Fit Scaler, store as attribute (so that I can reuse the transform method in the apply method)
        scaler_individual_vars = CustomScaler()
        self.fitted_individual_scaler = scaler_individual_vars.fit(individual_df)
        individual_df_scaled = self.fitted_individual_scaler.transform(individual_df)

        individual_df_scaled = individual_df_scaled.drop(self.fitted_individual_scaler.other_cols, axis=1)

        # INFO: The imputer is fitted only with the ml features, not the other cols
        if self.model == 'elasticnet':
            self.logger.log(f"        Fit linear imputer")
            individual_imputer = self._fit_linear_imputer(
                df=individual_df_scaled,
                num_imputation=num_imputation
            )
        elif self.model == 'randomforestregressor':
            self.logger.log(f"        Fit nonlinear imputer")
            individual_imputer = self._fit_nonlinear_imputer(
                df=individual_df_scaled,
                num_imputation=num_imputation
            )
        else:
            raise ValueError(f"Imputations for model {self.model} not implemented")

        return individual_imputer

    def prepare_country_df(self, df: pd.DataFrame, country_var_cols: list):
        """
        This method groups the individual-level data by country and year for the mac imputations. This is used
        in the fit and transform methods applied to the country variables

        Args:
            df:

        Returns:

        """
        # Store original index
        df_exploded = df.copy()
        df_exploded['original_index'] = df_exploded.index

        df_exploded[self.years_col] = df[self.years_col].apply(
            lambda x: list(x) if isinstance(x, tuple) else [x]
        )
        df_exploded = df_exploded.explode(self.years_col)

        # Group by country and year
        group_cols = [self.country_group_by, self.years_col]
        country_df = df_exploded.groupby(group_cols)[country_var_cols].first().reset_index()
        return country_df, group_cols, df_exploded

    def _apply_country_level_imputation(self, df, country_var_cols, num_imputation):
        """
        Apply the country-level imputation based on the fitted imputer.

        Args:
            df: DataFrame to apply imputation to.
            country_var_cols: List of country-level variable columns.
            num_imputation: The number of imputations for missing data.

        Returns:
            DataFrame with imputed country-level variables.
        """
        df_tmp = df.copy()
        country_df, group_cols, df_exploded = self.prepare_country_df(df=df_tmp, country_var_cols=country_var_cols)
        country_df_scaled = self.fitted_country_scaler.transform(country_df)

        # This has the parmeters of iterativeImputer (thus, the RFR imputer should have the same arguments)
        # Random state is handled during fitting I guess?
        country_array_imputed_scaled = self.country_imputer.transform(
            X=country_df_scaled[country_var_cols],
        )

        # Create DataFrame of imputed variables
        country_df_imputed_scaled = pd.DataFrame(
            country_array_imputed_scaled,
            columns=country_var_cols,
        )
        # Re-scale the variables
        country_df_imputed = self.fitted_country_scaler.inverse_transform(country_df_imputed_scaled)

        # Include the group_cols in country_df_imputed
        country_df_imputed[group_cols] = country_df[group_cols]

        # Merge back the imputed country-level data to the exploded DataFrame
        df_exploded = df_exploded.drop(columns=country_var_cols)
        other_columns = df_exploded.columns.drop("original_index")
        df_exploded = df_exploded.merge(country_df_imputed, on=group_cols, how='left')

        # individual vars are not affected by the country-level imputation
        individual_var_df = df[other_columns].copy()

        # isolate country-level columns
        df_exploded_country = df_exploded.drop(columns=other_columns)

        # Group country columns by the original index
        df_country_grouped = df_exploded_country.groupby(df_exploded_country["original_index"])
        # Perform the aggregation
        df_country_aggregated = df_country_grouped.agg("mean")
        df_merged = pd.concat([individual_var_df, df_country_aggregated], axis=1)
        assert df_merged.index.all() == df.index.all(), "Indices between merged and original df not matching"
        return df_merged

    def _apply_individual_level_imputation(self, df, individual_var_cols, num_imputation):
        """
        Apply the individual imputation based on the fitted imputer.

        Args:
            df: DataFrame to apply imputation to.
            individual_var_cols: List of country-level variable columns.
            num_imputation: The number of imputations for missing data.

        Returns:
            DataFrame with imputed country-level variables.
        """
        """
        Imputes missing values on the individual level.
        """
        individual_df = df[individual_var_cols]
        print(len(individual_df))

        # scale individual cols
        individual_df_scaled = self.fitted_individual_scaler.transform(individual_df)

        individual_array_imputed_scaled = self.individual_imputer.transform(
            X=individual_df_scaled[individual_var_cols],
        )

        # Merge
        individual_df_imputed_scaled = pd.DataFrame(individual_array_imputed_scaled, columns=individual_df.columns, index=individual_df.index)
        individual_df_imputed = self.fitted_individual_scaler.inverse_transform(individual_df_imputed_scaled)

        df = df.drop(columns=individual_var_cols)
        # Former problem is fixed -> feature order is valid in combined analyses
        df = pd.concat([individual_df_imputed, df], axis=1, join="outer")

        return df

    def _fit_linear_imputer(self, df: pd.DataFrame, num_imputation: int):
        """
        Applies linear imputations using the IterativeImputer from sklearn.
        In analysis with many features, we reduce the number of features used for imputation.
        """

        n_features = int(len(df.columns) * self.percentage_of_features)
        if n_features < self.n_features_thresh:
            n_features = None

        # Separate binary from continuous cols for estimator
        binary_cols = df.columns[(df.isin([0, 1]) | df.isna()).all(axis=0)].tolist()
        # Return the indices of the binary columns
        binary_col_indices = [df.columns.get_loc(col) for col in binary_cols]

        # Instantiate the custom estimator
        adaptive_estimator = AdaptiveImputerEstimator(
            regressor=Ridge(),
            classifier=SafeLogisticRegression(penalty="l2"),  # Ridge Penalty
            categorical_idx=binary_col_indices
        )

        # Set up the IterativeImputer with the custom estimator
        imputer = CustomIterativeImputer(
            estimator=adaptive_estimator,
            sample_posterior=self.sample_posterior,  # if sample posterior == False, apply PMM
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,
            categorical_idx=binary_col_indices,
            n_nearest_features=n_features,
            pmm_k=self.pmm_k,
        )

        # Fit the imputer on the data
        imputer.fit(df)

        return imputer

    def _fit_nonlinear_imputer(self, df: pd.DataFrame, num_imputation: int) -> NonLinearImputer:
        """
        Fits the NonLinearImputer on the training data.
        """
        imputer = NonLinearImputer(
            logger=self.logger,
            max_iter=self.max_iter,
            random_state=self.fix_rs + num_imputation,
            tree_max_depth=self.tree_max_depth
        )
        imputer.fit(df)
        return imputer


