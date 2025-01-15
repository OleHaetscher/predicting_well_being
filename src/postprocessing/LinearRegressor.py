import pandas as pd
import statsmodels.api as sm
from sklearn import clone
from sklearn.preprocessing import StandardScaler

from src.utils.DataSelector import DataSelector
from src.analysis.Imputer import Imputer
from src.utils.Logger import Logger


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
        meta_vars,
        num_features=6,
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
        self.meta_vars = meta_vars

        self.dataselector = DataSelector(
            self.var_cfg,
            self.df,
            self.feature_combination,
            self.crit,
            self.samples_to_include,
            self.meta_vars
        )

        self.logger = Logger(
            log_file=self.var_cfg["general"]["log_name"],
        )

        self.country_grouping_col = self.var_cfg["analysis"]["imputation"][
            "country_grouping_col"
        ]
        self.years_col = self.var_cfg["analysis"]["imputation"]["years_col"]

    @property
    def imputer(self) -> Imputer:
        """
        Creates and returns an instance of the Imputer class configured for the current analysis.

        The Imputer is initialized with settings derived from the configuration (`var_cfg`) and includes:
        - Logging through the logger instance.
        - Model-specific configurations like random state, convergence threshold, and imputation parameters.
        - Columns for grouping data (e.g., by country or year).
        As the Imputer depends on self.model which is defined in the subclass, we cannot set this in the __init__ method.

        Returns:
            Imputer: An Imputer instance initialized with the specified settings.
        """
        return Imputer(
            logger=self.logger,
            model="elasticnet",  # TODO does this work?
            fix_rs=self.var_cfg["analysis"]["random_state"],
            max_iter=40,  # test, otherwise 40   # self.var_cfg["analysis"]["imputation"]["max_iter"],
            num_imputations=1,
            conv_thresh=1,  # not relevant, only RFR
            tree_max_depth=1,  # not relevant, only RFR
            percentage_of_features=1,
            n_features_thresh=6,
            sample_posterior=False,
            pmm_k=5,
            country_group_by=self.country_grouping_col,
            years_col=self.years_col,
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
        # Mean imputation for missing values in features  # TODO Use linear imputer class? -> I should do this
        # Impute missing values with our custom linear imputer
        X = self.X.copy()
        imputer = clone(self.imputer)

        nan_counts_before = self.X.isnull().sum()

        imputer.fit(X, num_imputation=1)
        X_imputed = imputer.transform(X=X)

        nan_counts_after = X_imputed.isnull().sum()

        # Log or print the NaN counts for comparison
        print("NaN counts before imputation:")
        print(nan_counts_before)

        print("\nNaN counts after imputation:")
        print(nan_counts_after)

        self.y = self.y.loc[X_imputed.index]

        X_imputed = X_imputed.drop(columns=self.meta_vars, errors="ignore")

        # Standardize features to zero mean and unit variance
        scaler = StandardScaler()
        X_imputed_scaled = pd.DataFrame(scaler.fit_transform(X_imputed),
                                        columns=X_imputed.columns,
                                        index=X_imputed.index)

        # Add an intercept to the model
        X_imputed_scaled_intercept = sm.add_constant(X_imputed_scaled)

        # Fit OLS regression model
        model = sm.OLS(self.y, X_imputed_scaled_intercept).fit()

        # Print the summary of the regression results
        print()
        print()
        print()
        print("-----")
        print(f"Regression results for {self.feature_combination} - {self.crit}")
        print(model.summary())

        return model

