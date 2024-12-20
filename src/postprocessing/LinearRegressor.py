import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.utils.DataSelector import DataSelector


class LinearRegressor:
    """
    This class computes data-driven linear regression models with the best X features for a
    given analysis setting using statsmodels.
    """

    def __init__(
        self,
        var_cfg,
        df,
        feature_combination,
        crit,
        samples_to_include,
        processed_output_path,
        model_for_features,
        num_features=10,
    ):
        self.var_cfg = var_cfg
        self.processed_output_path = processed_output_path
        self.df = df

        self.feature_combination = feature_combination
        self.crit = crit
        self.samples_to_include = samples_to_include
        self.model_for_features = model_for_features  # RFR
        self.num_features = num_features

        self.datasets_included = None
        self.X = None
        self.y = None
        self.rows_dropped_crit_na = None

        self.dataselector = DataSelector(
            self.var_cfg,
            self.df,
            self.feature_combination,
            self.crit,
            self.samples_to_include
        )

    def get_regression_data(self):
        """

        Returns:

        """
        self.dataselector.select_samples()
        X = self.dataselector.select_features()
        self.y = self.dataselector.select_criterion()
        self.X = self.dataselector.select_best_features(
            df=X,
            root_path=self.processed_output_path,
            model=self.model_for_features,
            num_features=self.num_features,
        )

    def compute_regression_models(self):
        """
        Args:
            None

        Returns:
            model_results: A fitted statsmodels OLS regression results object
                           containing the model parameters, statistical tests, and summary.
        """
        # Mean imputation for missing values in features  # TODO Use linear imputer class?
        X = self.X.apply(lambda col: col.fillna(col.mean()), axis=0)

        self.y = self.y.loc[X.index]

        # Standardize features to zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        # Add an intercept to the model
        X_scaled_intercept = sm.add_constant(X_scaled)

        # Fit OLS regression model
        model = sm.OLS(self.y, X_scaled_intercept).fit()

        # Print the summary of the regression results
        print()
        print()
        print()
        print("-----")
        print(f"Regression results for {self.feature_combination} - {self.crit}")
        print(model.summary())

        ###### curiosity test
        if self.feature_combination == "srmc":
            # Add specified interaction terms
            interaction_terms = {
                'percentage_interactions - sleep_quality_mean': X_scaled['srmc_percentage_interactions'] * X_scaled['srmc_sleep_quality_mean'],
                'sleep_quality_mean - sleep_quality_max': X_scaled['srmc_sleep_quality_mean'] * X_scaled['srmc_sleep_quality_max'],
                'sleep_quality_mean - sleep_quality_min': X_scaled['srmc_sleep_quality_mean'] * X_scaled['srmc_sleep_quality_min'],
                'percentage_interactions - ftf_interactions': X_scaled['srmc_percentage_interactions'] * X_scaled['srmc_ftf_interactions'],
                # 'sleep_quality_mean - percentage_responses': X_scaled['srmc_sleep_quality_mean'] * X_scaled['srmc_percentage_responses'],
                'number_interactions - percentage_interactions': X_scaled['srmc_number_interactions'] * X_scaled['srmc_percentage_interactions'],
            }
            # Add interaction terms to the DataFrame
            for name, term in interaction_terms.items():
                X_scaled[name] = term

            # Add an intercept to the model
            X_scaled_intercept_ia = sm.add_constant(X_scaled)

            # Fit OLS regression model
            model = sm.OLS(self.y, X_scaled_intercept_ia).fit()

            # Print the summary of the regression results
            print("-----")
            print(f"Regression results for {self.feature_combination} - {self.crit}, interaction values ")
            print(model.summary())

    def store_regression_results(self):
        pass

