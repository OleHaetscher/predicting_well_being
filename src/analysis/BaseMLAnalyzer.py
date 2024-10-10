import gc
import inspect
import json
import os
import pickle
import threading
from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed, Memory
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    make_scorer,
    get_scorer,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.analysis.CustomScaler import CustomScaler
from src.analysis.Imputer import Imputer
from src.analysis.ShuffledGroupKFold import ShuffledGroupKFold
from src.utils.DataLoader import DataLoader
from src.utils.Logger import Logger
from src.utils.Timer import Timer

pd.options.mode.chained_assignment = None  # default='warn'


class BaseMLAnalyzer(ABC):
    """
    Abstract base class for the machine learning-based prediction procedure. This class serves as a template
    for the model specific class implementations. It encapsulates basic functionality of the repeated-nested
    cv procedure and the feature importance analysis that is model independent.
    #TODO: Adjust Documentation
    Attributes:
        output_dir: str, Specific directory where the result files of the machine learning analysis are stored.
            This depends on multiple parameters, such as the analysis_type (main / suppl), the study (ssc/mse),
            the esm-sample, the feature inclusion strategy, the prediction model, and in SSC the social situation
            variable. An example would be "../results/main/ssc/coco_int/scale_means/lasso/social_interaction".
                If the models are computed locally, the root directory is defined in the var_cfg.
                If the models are computed on a cluster, the root directory is defined in the SLURM script.
            Construction of further path components (e.g. "main/ssc/coco_int/scale_means/lasso/social_interaction")
            follow the same logic, both locally and on a cluster.
        model: str, prediction model for a given analysis, defined through the subclasses.
        best_models: Dict that collects the number of repetitions of the repeated nested CV as keys and a list
            of best models (grid_search.best_estimator_.named_steps["model"]) obtained in the nested CV as values
            during the machine learning analysis.
        repeated_nested_scores: Nested Dict that collects the number of repetitions as keys in the outer nesting,
            the metrics as keys in the inner nesting and the values obtained in the hold-out test sets in one outer
            fold as a list of values in the inner nesting.
                Example: {"rep_1":{"r2":[0.05, 0.02, 0.03]}}
        shap_values: Nested Dict that collects the SHAP values summarized across outer folds per repetition seperated
            for the train and the test set. SHAP values are of shape (n_samples, n_features).
                Example: {"train": {"rep_1": [n_samples, n_features]}}
        shap_ia_values: Nested Dict that collects aggregates of SHAP interaction values summarized across outer folds
            and repetitions separated for the train and test set. Because storing the raw interaction SHAP values would
            have been to memory intensive (a n_samples x n_samples x n_features tensor), we already calculated
            meaningful aggregations on the cluster (e.g., summarizing across).
                Example: {"train": {"agg_ia_persons": {"age_clean": { "age_clean": 112.43, 1.59}}}}
        var_cfg: YAML var_cfg determining certain specifications of the analysis. Is used on the cluster and locally.
            In contrast to the other classes, the var_cfg gets updated dynamically on certain conditions (e.g.,
            number of inner and outer folds is always set to 10 when running on a cluster to prevent errors caused
            by manual adjusting of the var_cfg for local testing).
        sample_weights: [pd.Series, None], The individual reliabilities calculated in the MultilevelModeling Class
            used to weight the individual samples in the machine learning based prediction procedure. Is None, if
            suppl_type != weighting_by_rel. If suppl_type == weighting_by_rel, the sample weights are loaded from
            its directory and used in the repeated nested CV using sklearns metadata_routing.
        pipeline: sklearn.pipeline.Pipeline, pipeline defining the steps that are applied inside the repeated-nested
            CV. This includes preprocessing (i.e., scaling), recursive feature elimination (if feature inclusion
            strategy == feature_selection) and the prediction model
        X: pd.df, All features for a given ESM-sample and feature inclusion strategy.
        y: pd.Series, All criteria for a given ESM-sample (thus, the individual reactivity estimates).
    """

    @abstractmethod
    def __init__(
        self,
        var_cfg,
        spec_output_path,
        df,
    ):
        """
        Constructor method of the BaseMLAnalyzer Class.

        Args:
            var_cfg: YAML var_cfg determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        self.var_cfg = var_cfg
        self.spec_output_path = spec_output_path

        self.crit = self.var_cfg["analysis"]["params"]["crit"]
        self.feature_combination = self.var_cfg["analysis"]["params"]["feature_combination"]
        self.samples_to_include = self.var_cfg["analysis"]["params"]["samples_to_include"]

        self.logger = Logger(log_dir=self.spec_output_path, log_file=self.var_cfg["general"]["log_name"])
        self.timer = Timer(self.logger)
        self.data_loader = DataLoader()

        # Methods that get clocked
        self.repeated_nested_cv = self.timer._decorator(self.repeated_nested_cv)
        self.nested_cv = self.timer._decorator(self.nested_cv)
        self.summarize_shap_values_outer_cv = self.timer._decorator(self.summarize_shap_values_outer_cv)
        self.impute_datasets_for_fold = self.timer._decorator(self.impute_datasets_for_fold)
        self.clocked_grid_search_fit = self.timer._decorator(self.clocked_grid_search_fit)

        # Data
        self.df = df
        self.X = None
        self.y = None
        self.binary_cols = None
        self.continuous_cols = None

        # Results
        self.best_models = {}
        self.repeated_nested_scores = {}
        self.shap_results = {'shap_values': {}, "base_values": {}, "data": {}}
        self.shap_ia_values = {}  # ggfs adjust
        self.lin_model_coefs = {}  # ggfs adjust

        # Defined in subclass
        self.model = None

        # CV parameters
        self.pipeline = None
        self.datasets_included = None
        self.num_inner_cv = self.var_cfg["analysis"]["cv"]["num_inner_cv"]
        self.num_outer_cv = self.var_cfg["analysis"]["cv"]["num_outer_cv"]
        self.num_reps = self.var_cfg["analysis"]["cv"]["num_reps"]
        self.num_imputations = self.var_cfg["analysis"]["imputation"]["num_imputations"]
        self.id_grouping_col = self.var_cfg["analysis"]["cv"]["id_grouping_col"]

    @property
    def model_name(self):
        """Get a string repr of the model name and sets it as class attribute (e.g., "lasso")."""
        return self.model.__class__.__name__.lower()

    @property
    def imputer(self):
        return Imputer(
            model=self.model_name,
            fix_rs=self.var_cfg["analysis"]["random_state"],
            max_iter=self.var_cfg["analysis"]["imputation"]["max_iter"],
            num_imputations=self.var_cfg["analysis"]["imputation"]["num_imputations"],
            n_jobs_imputation_columns=self.var_cfg["analysis"]["parallelize"]["imputation_columns_n_jobs"],
            conv_thresh=self.var_cfg["analysis"]["imputation"]["conv_thresh"],
            logger=self.logger,
        )

    @property
    def hyperparameter_grid(self):
        """Set hyperparameter grid defined in var_cfg for the specified model as class attribute."""
        return self.var_cfg["analysis"]["model_hyperparameters"][self.model_name]

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the var_cfg."""
        for method in self.var_cfg["analysis"]["methods_to_apply"]:
            if method not in dir(BaseMLAnalyzer):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            log_message = f"  Executing {getattr(self, method).__name__}"
            # self.logger.log(log_message)
            print(log_message)
            getattr(self, method)()

    def select_samples(self):
        """
        This method selects the samples based on the given combination using the indices that correspond
        to the samples (e.g., cocoesm_1)
        """
        if self.samples_to_include in ["selected", "control"]:
            datasets_included = self.var_cfg["analysis"]["feature_sample_combinations"][self.feature_combination]
            datasets_included_filtered = [dataset for dataset in datasets_included if dataset in
                                          self.var_cfg["analysis"]["crit_available"][self.crit]]
            self.datasets_included = datasets_included_filtered
            df_filtered = self.df[self.df.index.to_series().apply(
                lambda x: any(x.startswith(sample) for sample in self.datasets_included))]

            # for "sens" and "selected", remove samples without sensing data
            if self.samples_to_include == "selected" and self.feature_combination == "sens":
                sens_columns = [col for col in self.df.columns if col.startswith("sens_")]
                df_filtered = self.df[self.df[sens_columns].notna().any(axis=1)]

            # df_filtered = df_filtered.sample(n=300)  # just for testing
            self.df = df_filtered

        else:  # include all datasets
            self.datasets_included = self.var_cfg["analysis"]["feature_sample_combinations"]["all_in"]

    def select_features(self):
        """
        This method loads the machine learning features (traits) according to the specifications in the var_cfg.
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "X".
        ADJUST!!!!
        """
        selected_columns = [self.id_grouping_col]
        if self.samples_to_include in ["all", "selected"]:
            feature_prefix_lst = self.feature_combination.split("_")
        elif self.samples_to_include == "control":
            feature_prefix_lst = ["pl"]
        else:
            raise ValueError

        # always include grouping id
        for feature_cat in feature_prefix_lst:
            for col in self.df.columns:
                if col.startswith(feature_cat):
                    selected_columns.append(col)

        X = self.df[selected_columns].copy()

        setattr(self, "X", X)

    def select_criterion(self):
        """
        This method loads the criterion (reactivities, either EB estimates of random slopes or OLS slopes)
        according to the specifications in the var_cfg.
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "y".
        """
        y = self.df[f'crit_{self.crit}']
        assert len(self.X) == len(
            y
        ), f"Features and criterion differ in length, len(X) == {len(self.X)}, len(y) == {len(y)}"

        setattr(self, "y", y)

    def separate_binary_continuous_cols(self):
        """
        This function determines which features of the current data are continuous (i.e., containing not only 0/1 per column)
        and which features are binary (i.e., only containing 0/1 per column). This may also includes NaNs.
        It sets the resulting column lists as class attributes

        Returns:

        """
        data = self.X.copy()
        binary_cols = data.columns[(data.isin([0, 1]) | data.isna()).all(axis=0)]
        continuous_cols = [col for col in data.columns if col not in binary_cols]  # ordered as in df
        setattr(self, "binary_cols", binary_cols)
        setattr(self, "continuous_cols", continuous_cols)

    def initial_info_log(self):
        """
        Create log message at the beginning of the analysis. This helps to identify logs, validate the parameter passing
        of the SLURM script and provides further sanity-checking information.
        """
        self.logger.log("----------------------------------------------")

        self.logger.log("Global analysis params")
        self.logger.log(f"    Prediction model: {self.model_name}")
        self.logger.log(f"    Criterion: {self.crit}")
        self.logger.log(f"    Feature combination: {self.feature_combination}")
        self.logger.log(f"    Samples to included: {self.samples_to_include}")
        self.logger.log(f"    Actual datasets: {self.datasets_included}")

        self.logger.log("CV params")
        self.logger.log(f"    N repetitions: {self.num_reps}")
        self.logger.log(f"    N outer_cv: {self.num_outer_cv}")
        self.logger.log(f"    N inner_cv: {self.num_inner_cv}")
        self.logger.log(f"    N imputations: {self.num_imputations}")
        self.logger.log(f'    Model hyperparameters: {self.hyperparameter_grid}')
        self.logger.log(f'    N hyperparameter combinations: {len(list(product(*self.hyperparameter_grid.values())))}')
        self.logger.log(f'    Optimization metric: {self.var_cfg["analysis"]["scoring_metric"]["inner_cv_loop"]["name"]}')
        self.logger.log(f'    Max iter imputations: {self.var_cfg["analysis"]["imputation"]["max_iter"]}')
        self.logger.log(f'    Number of hyperpar')


        self.logger.log("Parallelization params")
        self.logger.log(f'    N jobs inner_cv: {self.var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"]}')
        self.logger.log(f'    N jobs shap: {self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"]}')
        self.logger.log(f'    N jobs imputation runs: {self.var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"]}')
        self.logger.log(f'    N jobs imputation columns: {self.var_cfg["analysis"]["parallelize"]["imputation_columns_n_jobs"]}')
        if self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]:
            self.logger.log(f'    N jobs shap_ia_values: {self.var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"]}')

        self.logger.log("Data params")
        self.logger.log(f"    Number of rows: {self.X.shape[0]}")
        self.logger.log(f"    Number of cols: {self.X.shape[1]}")
        na_counts = self.X.isna().sum()
        for column, count in na_counts[na_counts > 0].items():
            self.logger.log(f"      Number of NaNs in column {column}: {count}")
        self.logger.log("----------------------------------------------")

    def drop_zero_variance_cols(self):
        """
        This method checks if there are column in the features (self.X) that have no variance (i.e., only one value and np.nan),
        removes the values from the features (because this would slow down the computation), and logs which features
        were dropped
        """
        zero_variance_cols = []
        for column in self.X.columns:
            unique_values = self.X[column].nunique(dropna=True)
            if unique_values == 1:
                zero_variance_cols.append(column)
                self.logger.log(f"      Dropping column '{column}' due to zero variance (only one unique value or NaNs).")
        if zero_variance_cols:
            self.X.drop(columns=zero_variance_cols, inplace=True)
            self.logger.log(f"      {len(zero_variance_cols)} column(s) with zero variance dropped: {zero_variance_cols}")
        else:
            self.logger.log("      No columns with zero variance found.")

    def fit(self, X, y):
        """Scikit-Learns "Fit" method of the machine learning model, model dependent, implemented in the subclasses."""
        self.model.fit(X, y)

    def predict(self, X):
        """Scikit-Learns "Predict" method of the machine learning model, model dependent, implemented in the subclasses."""
        return self.model.predict(X)

    def create_pipeline(self):
        """
        This function creates a pipeline with preprocessing steps (e.g., scaling X, scaling y) and
        the estimator used in the repeated nested CV procedure. It sets the pipeline as a class attribute.
        """
        preprocessor = self.get_custom_scaler(data=self.X)
        target_scaler = StandardScaler()

        # Wrap your model (preprocessors are only applied to X, not to)
        model_wrapped = TransformedTargetRegressor(
            regressor=self.model,
            transformer=target_scaler
        )
        if self.var_cfg["analysis"]["cv"]["warm_start"]:
            model_wrapped.regressor.set_params(warm_start=True)

        # cache preprocessing steps to save time
        if self.var_cfg["analysis"]["cv"]["cache_pipe"]:
            cache_folder = "./cache_directory"
            memory = Memory(cache_folder, verbose=0)
        else:
            memory = None

        # create pipeline
        pipe = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", model_wrapped),
            ],
            memory=memory
        )
        setattr(self, "pipeline", pipe)

    def get_custom_scaler(self, data):
        """
        This function creates a custom scaler that scales only continuous columns. Feature order is preserved.

        Args:
            data: df, containing the features for a given analysis setting (binary and continuous)

        Returns:
            preprocessor: CustomScaler object that scales only continuous columns. Binary columns are still 0/1.

        """
        continuous_cols = self.continuous_cols.copy()
        if self.id_grouping_col in continuous_cols:
            continuous_cols.remove(self.id_grouping_col)
        preprocessor = CustomScaler(cols_to_scale=continuous_cols)
        return preprocessor

    def nested_cv(
        self,
        rep=None,
        X=None,
        y=None,
        fix_rs=None,
        dynamic_rs=None,
    ):
        """
        This method represents the fundamental aspect of the BassMLAnalyzer class.
        This function performs nested cross-validation for a given partition of the total data in a train and test set.
        In the current analysis, this is only used in combination with the "repeated_nested_cv" method, repeating
        the CV procedure with 10 different data partitions (as defined by the dynamic random state that is passed
        to this method).
        In all specifications, this method
            - iterates over the CV splits
            - splits X and y in train and test data accordingly
            - performs GridSearchCV on train data (which repeatedly splits the train data into train/validation data)
            - evaluates the best performing models in GridSearchCV on the test data for multiple metrics
            - summarizes the model-specific SHAP calculations across outer folds
        Depending on the specification, this method might
            - enable metadata routing and aligns the sample_weights with the features ("weighting_by_rel")
            - implement a sanity check if the feature selection (RFECV) worked as expected
            - contain SHAP interaction values (only in certain analyses and if model == rfr), empty Dicts otherwise

        Args:
            rep: int, representing the number of repetitions of the nested CV procedure, for 10x between 1 and 10
            X: df, features for the machine learning analysis according to the current specification
            y: pd.Series, criteria for the machine learning analysis according to the current specification
            fix_rs: int, random_state parameter, which should remain the same for all analyses
            dynamic_rs: int, dynamic random_state parameter that changes across repetitions (so that the splits into
                train and test set differ between different repetitions)
            use_sample_weights: bool, true or false, indicates if sample_weights are used in the prediction process
            sample_weights: pd.Series or None, containing the sample weights in "weighting_by_rel"

        Returns:
            nested_scores_rep: Dict that collects the metrics evaluated on the test sets as keys and the
                values obtained in each test set as a list of values.
                Example: {"r2":[0.05, 0.02, 0.03], "spearman": [.20, .15, .25]}
            np.mean(nested_score): Mean of the scores in the outer test sets for the current data partition. Only
                displayed for the metric that was used to evaluated GridSearchCV (currently MSE)
            np.std(nested_score): SD of the scores in the outer test sets for the current data partition. Only
                displayed for the metric that was used to evaluated GridSearchCV (currently R²)
            ml_models: list of the best models and their hyperparameter var_cfguration obtained in the GridSearchCV
                (grid_search.best_estimator_.named_steps["model"])
            shap_values_train: ndarray, SHAP values obtained for the train set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_values_test: ndarray, SHAP values obtained for the test set, summarized across outer folds.
                Of shape (n_samples x n_features).
            shap_ia_values_train: [None, ndarray], SHAP interaction values obtained for the train set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the var_cfg, None otherwise.
            shap_ia_values_test: [None, ndarray], SHAP interaction values obtained for the test set, summarized
                across outer folds. Of shape (n_samples x n_features x n_features). Only if model == rfr and
                calc_ia_values is enabled in the var_cfg, None otherwise.
        """
        # Set X and y
        if X is None:
            X = self.X
        if y is None:
            y = self.y

        # Align X and y
        X, y = X.align(y, axis=0, join="inner")

        # Create splits
        inner_cv = ShuffledGroupKFold(n_splits=self.num_inner_cv, random_state=fix_rs, shuffle=True)
        outer_cv = ShuffledGroupKFold(n_splits=self.num_outer_cv, random_state=dynamic_rs, shuffle=True)

        # Containers to hold the results
        ml_pipelines = []
        ml_model_params = []
        nested_scores_rep = {}
        X_test_imputed_lst = []

        # Loop over outer cross-validation splits
        for cv_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y, groups=X[self.id_grouping_col])):

            ml_pipelines_sublst = []
            ml_model_params_sublst = []
            nested_scores_rep[f"outer_fold_{cv_idx}"] = {}

            # Convert indices and select data
            train_indices = X.index[train_index]
            test_indices = X.index[test_index]
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]

            # Ensure indices match between X and y and group splitting works
            assert (X_train.index == y_train.index).all(), "Indices between train data differ"
            assert (X_test.index == y_test.index).all(), "Indices between test data differ"
            assert len(set(X_train[self.id_grouping_col]).intersection(set(X_test[self.id_grouping_col]))) == 0, \
                "Grouping did not work as expected"

            # TODO Remove, just for testing
            y_test = np.random.randn(len(y_test))

            # Create imputed datasets and save the test datasets for SHAP computations
            X_train_imputed_sublst, X_test_imputed_sublst = self.impute_datasets_for_fold(X_train=X_train, X_test=X_test)
            X_test_imputed_lst.append(X_test_imputed_sublst)

            scoring_inner_cv = get_scorer(self.var_cfg["analysis"]["scoring_metric"]["inner_cv_loop"]["name"])

            # Perform GridSearchCV for each imputation and aggregate results
            for imputed_idx in range(self.num_imputations):
                grid_search = GridSearchCV(
                    estimator=self.pipeline,
                    param_grid=self.hyperparameter_grid,
                    cv=inner_cv,
                    scoring=scoring_inner_cv,
                    refit=True,
                    verbose=self.var_cfg["analysis"]["cv"]["verbose_inner_cv"],
                    error_score="raise",
                    n_jobs=self.var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"],
                )

                nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{imputed_idx}"] = {}

                # This would be cleaner with metadata routing, but this works ATM
                groups = X_train_imputed_sublst[imputed_idx].pop('other_unique_id')
                X_train_current = X_train_imputed_sublst[imputed_idx]
                X_test_current = X_test_imputed_sublst[imputed_idx]
                le = LabelEncoder()
                groups_numeric = le.fit_transform(groups)

                # Fit grid search and clock execution
                grid_search = self.clocked_grid_search_fit(grid_search=grid_search,
                                                           X_train=X_train_current,
                                                           y_train=y_train,
                                                           groups=groups_numeric)

                # Append model (elasticnet) or model params (RFR)
                if self.model_name == "elasticnet":
                    ml_model_params_sublst.append(grid_search.best_estimator_.named_steps["model"].regressor_)
                else:
                    ml_model_params_sublst.append(grid_search.best_estimator_.named_steps["model"].regressor_.get_params())
                ml_pipelines_sublst.append(grid_search.best_estimator_)

                # Evaluate the model on the imputed test set and store the metrics
                scores = self.get_scores(grid_search, X_test_current, y_test)
                for metric, score in scores.items():
                    nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{imputed_idx}"][metric] = score

                # Free up memory
                del grid_search
                del X_train_current, X_test_current

            # Append sublists to list
            ml_model_params.append(ml_model_params_sublst)
            ml_pipelines.append(ml_pipelines_sublst)

        # Summarize SHAP values and return all results
        (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,
        ) = self.summarize_shap_values_outer_cv(
            X_test_imputed_lst=X_test_imputed_lst,
            X=X,
            y=y,
            pipelines=ml_pipelines,
            outer_cv=outer_cv
        )

        del ml_pipelines
        gc.collect()

        return (
            nested_scores_rep,
            ml_model_params,
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,
        )

    def clocked_grid_search_fit(self, grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series):
        """
        Clocked version of gridsearch.fit

        Args:
            grid_search:
            X_train:
            y_train:
            groups:

        Returns:
            GridSearchCV: Fitted gridsearch object
        """
        grid_search.fit(X_train, y_train, groups=groups)
        return grid_search

    def impute_datasets_for_fold(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        This function creates num_imputations datasets in parallel using joblib.

        Args:
            X_train: Training data
            X_test: Test data

        Returns:
            tuple(list[pd.DataFrame], list[pd.DataFrame]): tuple containing a list of imputed train and test datasets
                that both have lengths of self.num_imputations
        """
        # Ensure that the grouping column is not in the numeric data
        X_train_numeric = X_train.select_dtypes(include="number")
        X_test_numeric = X_test.select_dtypes(include="number")

        # Define the parallel imputation function
        def impute_single_dataset(i):
            # self.log_thread()
            # Clone the imputer to avoid shared state
            imputer = clone(self.imputer)

            # Fit the imputer on the training data
            imputer.fit(X_train_numeric)
            # Transform both training and test data
            X_train_imputed = imputer.transform(X_train_numeric, num_imputation=i)
            X_test_imputed = imputer.transform(X_test_numeric, num_imputation=i)

            # Concatenate non-numeric columns if necessary
            X_train_imputed = pd.concat([X_train_imputed, X_train[self.id_grouping_col]], axis=1)

            return X_train_imputed, X_test_imputed

        # Run the imputation in parallel
        results = Parallel(n_jobs=self.var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"])(
            delayed(impute_single_dataset)(i) for i in range(self.num_imputations)
        )

        # Unpack the results
        X_train_imputed_sublst, X_test_imputed_sublst = zip(*results)
        return list(X_train_imputed_sublst), list(X_test_imputed_sublst)

    def get_scores(self, grid_search, X_test, y_test):
        """
        This method generates scoring functions and evaluates the best model on the test set,
        verifying that the scaling operations were applied to X_test and y_test.

        Args:
            grid_search: The trained grid search object with the best estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            scores: A dictionary with the calculated scores for each metric.
        """
        # Define the scoring functions
        scoring_functions = {
            "neg_mean_squared_error": get_scorer("neg_mean_squared_error"),
            "r2": get_scorer("r2"),
            "spearman": make_scorer(self.spearman_corr),  # Assuming self.spearman_corr is defined
        }

        scorers = {
            metric: scoring_functions[metric]
            for metric in self.var_cfg["analysis"]["scoring_metric"]["outer_cv_loop"]
        }

        # Evaluate the best model on the test set
        # Note: This handles the crit scaling internally using "y_test" and "best_estimator"
        scores = {
            metric: scorer(grid_search.best_estimator_, X_test, y_test)
            for metric, scorer in scorers.items()
        }

        return scores

    def compute_shap_for_fold(
        self,
        num_cv_,
        num_imputation,
        pipeline,
        index_mapping,
        num_test_indices,
        X_test,
        all_features,
    ):
        """
        Parallelization implementation of the method "summarize_shap_values_outer_cv". This enables parallel SHAP
        calculations for outer_folds if specified in the var_cfg.
        Note: We only compute this for the test set.

        Args:
            num_cv_: int, indicating a certain run of the outer CV loop
            pipeline: Pipeline object, the best performing pipeline in the GridSearchCV associated with num_cv_
            index_mapping: Mapping from numerical indices to variable indices for unambiguous feature assignment
            num_test_indices: Numeric indices of the features that are in the test set in the outer loop "num_cv_"
            X: df, features for the machine learning-based prediction
            all_features: Index object (X.columns), representing the feature names

        Returns:
            train_shap_template: ndarray, containing the SHAP values of the train set for the outer fold "num_cv_"
                Shape: (n_features x n_samples), array values that represent samples that were in the test set
                in the current outer fold are all zero.
            test_shap_template: ndarray, containing the SHAP values of the test set for the outer fold "num_cv_"
                Shape: (n_features x n_samples), array values that represent samples that were in the train set
                in the current outer fold are all zero.
            num_train_indices[num_cv_]: Numerical indices for the samples in the train set for the current outer fold
            num_test_indices[num_cv_]: Numerical indices for the samples in the test set for the current outer fold
            train_ia_shap_template: [None, ndarray], containing the SHAP interaction values of the train set for
                the current outer fold "num_cv_". Shape: (n_features x n_features x n_samples), array values that
                represent samples that were in the test set in the current outer fold are all zero.
                Only defined if calc_ia_values is specific in the var_cfg and model == 'rfr', None otherwise
            test_ia_shap_template: [None, ndarray], containing the SHAP interaction values of the test set for
                the current outer fold "num_cv_". Shape: (n_features x n_features x n_samples), array values that
                represent samples that were in the train set in the current outer fold are all zero.
                Only defined if calc_ia_values is specific in the var_cfg and model == 'rfr', None otherwise
        """
        (
            shap_values_test,
            base_values_test,
            # data_test,
            features_test,
        ) = self.calculate_shap_values(X_test, pipeline)

        if (
            self.model_name == "randomforestregressor"
            and self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]
        ):
            shap_ia_values_test = self.calculate_shap_ia_values(X_test, pipeline)

        else:
            test_ia_shap_template = None

        return (
            shap_values_test,
            base_values_test,
            X_test,
            num_test_indices[num_cv_],
            num_cv_,
            test_ia_shap_template,
        )

    def summarize_shap_values_outer_cv(self,
                                       X_test_imputed_lst: list[list[pd.DataFrame]],
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       pipelines: list[list[sklearn.pipeline.Pipeline]],
                                       outer_cv):
        """
        Ths function summarizes the shap values of the repeated nested resampling scheme as described by
        Scheda & Diciotti (2022). We only use the SHAP values from the test sets.
        Thus, it takes the individual shap values of the test sets (when doing a 10x10 CV, a sample is 9 times
        in the train set and only 1 time in the test set, so we have only 1 test set shap value per individual sample
        in one repetition).
        It does this by defining a template a zeros of the shap of the SHAP values (n_features x n_samples) and
        accumulates the SHAP values computed in the outer_folds in this template.

        Args:
            X_test_imputed_lst: List of lists containing dataframes. The first list index corresponds to the outer_fold,
                the second list index corresponds to the imputations.Thus, X_train_imputed_lst[0][1] would correspond
                to outer_fold==0 and num_imputation==1
            X: Raw dataframe that contains all samples (training and test)
            y: Raw criterion
            pipelines: List of Pipeline objects, containing preprocessing steps and estimators that obtained the
                best performance in the GridSearchCV in "nested_cv"
            outer_cv: KFold object used in "nested_cv" to repeatedly split the data into train and test set

        Returns:
            avg_shap_values_train: ndarray, containing the mean SHAP values of the train set (n_features x n_samples)
            test_shap_values: ndarray, containing the SHAP values of the test set (n_features x n_samples)
            avg_ia_shap_values_train: [None, ndarray], containing the mean SHAP interaction values of the train set
                (n_features x n_features x n_samples) if calc_ia_values is specific in the var_cfg and model == 'rfr',
                None otherwise
            ia_test_shap_values: [None, ndarray], containing the SHAP interaction values of the test set
                (n_features x n_features x n_samples) if calc_ia_values is specific in the var_cfg and model == 'rfr',
                None otherwise
        """
        print('---------------------')
        print('Calculate SHAP values')
        # Create a mapping from numerical indices to variable indices
        index_mapping = dict(enumerate(X.index))
        # Get numerical indices of the samples in the outer cv
        num_test_indices = [test for train, test in outer_cv.split(X, y, groups=X.other_unique_id)]
        X = X.drop(columns=[self.id_grouping_col], axis=1)
        all_features = X.columns

        # Set up the array to store the results -> 3D arrays (rows x features x imputations)
        # base values may vary across samples
        rep_shap_values = np.zeros((X.shape[0], X.shape[1], self.num_imputations), dtype=np.float32)
        rep_base_values = np.zeros((X.shape[0], self.num_imputations), dtype=np.float32)  # we get different base values per fold
        rep_data = np.zeros((X.shape[0], X.shape[1], self.num_imputations), dtype=np.float32)

        # Compute and aggregate SHAP values for one fold
        results = [
            [
                self.compute_shap_for_fold(
                    num_cv_=num_cv_,
                    pipeline=pipelines[num_cv_][num_imputation],
                    index_mapping=index_mapping,
                    num_test_indices=num_test_indices,
                    X_test=X_test_imputed_lst[num_cv_][num_imputation],
                    num_imputation=num_imputation,
                    all_features=all_features,
                )
                for num_imputation in range(self.num_imputations)
            ]
            for num_cv_, _ in enumerate(pipelines)
        ]

        # Aggregate results from all folds and imputations
        for num_cv_, fold_results in enumerate(results):
            for num_imputation, (
                    shap_values_template,
                    base_values_template,
                    # data_template,
                    X_test,
                    test_idx,
                    _,
                    _,
            ) in enumerate(fold_results):
                # Aggregate results for the current fold and imputation
                rep_shap_values[test_idx, :, num_imputation] += shap_values_template.astype(np.float32)
                rep_base_values[test_idx, num_imputation] += base_values_template.flatten().astype(np.float32)
                rep_data[test_idx, :, num_imputation] += X_test.astype(np.float32)

        # We need to divide the base values, because we get base_values in every outer fold
        # Test if the SHAP assumptions hold with this procedure and plotting is possible
        rep_base_values = rep_base_values / self.num_outer_cv

        # TODO adjust
        if (
            self.model_name == "randomforestregressor"
            and self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]
        ):
            ia_test_shap_values = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            # Aggregate results from all folds
            for (
                _,
                _,
                _,
                test_idx,
                _,
                test_ia_shap_template,
            ) in results:
                ia_test_shap_values[test_idx, :, :] += test_ia_shap_template
        else:
            ia_test_shap_values = None

        return (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,
        )

    def repeated_nested_cv(self):
        """
        This function performs the nested cross-validation repeatedly using different data partitions into train
        and test sets. It does so by repeatedly executing the nested cv function using different
        dynamic random states that control the train-test partitioning. It also updates the class attributes
        with the corresponding analysis results.
        It further summarizes the SHAP values across repetitions and the SHAP interaction values (if specified)
        into meaningful aggregates (both steps are done on the cluster to save resources and memory).
        """
        X = self.X.copy()
        y = self.y.copy()
        fix_random_state = self.var_cfg["analysis"]["random_state"]

        # Execute the nested cross-validation procedure
        results = (
            self.nested_cv(
                rep=rep,
                X=X,
                y=y,
                fix_rs=fix_random_state,
                dynamic_rs=fix_random_state + rep,
            )
            for rep in range(self.num_reps)  # 0 - 9
        )
        for rep, (
            nested_scores_rep,
            best_models,
            rep_shap_values,
            rep_base_values,
            rep_data,
            shap_ia_values_test,
        ) in enumerate(results):
            self.best_models[f"rep_{rep}"] = best_models
            self.shap_results["shap_values"][f"rep_{rep}"] = rep_shap_values
            self.shap_results["base_values"][f"rep_{rep}"] = rep_base_values
            self.shap_results["data"][f"rep_{rep}"] = rep_data
            self.repeated_nested_scores[f"rep_{rep}"] = nested_scores_rep
            print()

            if (
                self.model_name == "randomforestregressor"
                and self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]
            ):  # TODO Adjust after main analysis
                self.summarize_shap_interaction_values(
                    shap_ia_values=shap_ia_values_test, columns=X.columns, dataset="test"
                )

        for rep in range(len(self.repeated_nested_scores)):
            print(f"scores for rep {rep}: ", self.repeated_nested_scores[f"rep_{rep}"])

    def summarize_shap_interaction_values(self, shap_ia_values, columns, dataset):
        """
        This function is used for summarizing the raw interactions values of shape (n_samples, n_features, n_features).
        1) It summarizes the raw and the absolute shap values across samples
        2) It identifies the most influential interaction per person and stores a) the variables and
           b) the pairs that interacts most in two dictionaries
        3) It summarizes the interaction values across features and samples, so that for each feature there
           is a score reflecting a "global" interaction value
        All results are stored as key:value pairs in the attribute self.shap_ia_values

        Args:
            shap_ia_values: ndarray, containing the IA values, Shap (n_samples x n_features xn_features)
            columns: pd.Index, Features corresponding to n_features in shap_ia_values
            dataset: str, "train" or "test"
        """
        # Rescale all shap_ia_values -> + 1 Mio -> is reverted in the ShapValueAnalyzer
        # TODO: do not do this
        shap_ia_values = shap_ia_values * 1000000

        # 1) Summarize across persons
        ia_across_persons = shap_ia_values.mean(0)
        ia_across_persons_df = pd.DataFrame(
            ia_across_persons, index=columns, columns=columns
        )
        ia_across_persons_dct = {
            col: {
                row: value for row, value in ia_across_persons_df[col].items()
            }  # keep the main effects
            for col in ia_across_persons_df.columns
        }
        abs_ia_across_persons = np.abs(shap_ia_values).mean(0)
        abs_ia_across_persons_df = pd.DataFrame(
            abs_ia_across_persons, index=columns, columns=columns
        )
        abs_ia_across_persons_dct = {
            col: {
                row: value for row, value in abs_ia_across_persons_df[col].items()
            }  # keep the main effects
            for col in abs_ia_across_persons_df.columns
        }

        # 2) Most influential interactions
        mask = np.eye(len(columns), dtype=bool)
        most_influential_ia_per_person = np.array(
            [
                np.unravel_index(
                    np.nanargmax(
                        np.where(mask, np.nan, np.abs(shap_ia_values[i, :, :]))
                    ),
                    (len(columns), len(columns)),
                )
                for i in range(shap_ia_values.shape[0])
            ]
        )
        most_influential_interaction_df = pd.DataFrame(
            [
                (columns[row], columns[col])
                for row, col in most_influential_ia_per_person
            ],
            columns=["Feature 1", "Feature 2"],
        )
        feature_counts = most_influential_interaction_df.apply(
            pd.Series.value_counts
        ).sum(axis=1)
        feature_count_dict = feature_counts.sort_values(ascending=False).to_dict()
        unordered_pairs = most_influential_interaction_df.apply(
            lambda row: tuple(sorted([row["Feature 1"], row["Feature 2"]])), axis=1
        )
        unordered_pairs_as_strings = [
            str(pair) for pair in unordered_pairs
        ]  # Convert tuples to strings
        pair_counts_dct = pd.Series(unordered_pairs_as_strings).value_counts().to_dict()

        # 3) Summarize across persons and features
        agg_ia_across_persons_ias = np.nanmean(
            np.where(mask, np.nan, shap_ia_values), axis=(0, 1)
        )
        agg_ia_across_persons_ias_df = pd.DataFrame(
            agg_ia_across_persons_ias, index=columns, columns=["agg_ia_val"]
        )
        agg_ia_across_persons_ias_dct = agg_ia_across_persons_ias_df[
            "agg_ia_val"
        ].to_dict()
        abs_agg_ia_across_persons_ias = np.nanmean(
            np.abs(np.where(mask, np.nan, shap_ia_values)), axis=(0, 1)
        )
        abs_agg_ia_across_persons_ias_df = pd.DataFrame(
            abs_agg_ia_across_persons_ias, index=columns, columns=["agg_ia_val"]
        )
        abs_agg_ia_across_persons_ias_dct = abs_agg_ia_across_persons_ias_df[
            "agg_ia_val"
        ].to_dict()

        # Set the aggregates in the class attribute Dict
        self.shap_ia_values[dataset]["agg_ia_persons"] = ia_across_persons_dct
        self.shap_ia_values[dataset]["abs_agg_ia_persons"] = abs_ia_across_persons_dct
        self.shap_ia_values[dataset][
            "most_influential_ia_features"
        ] = feature_count_dict
        self.shap_ia_values[dataset]["most_influential_ia_pairs"] = pair_counts_dct
        self.shap_ia_values[dataset][
            "agg_ia_persons_ias"
        ] = agg_ia_across_persons_ias_dct
        self.shap_ia_values[dataset][
            "abs_agg_ia_persons_ias"
        ] = abs_agg_ia_across_persons_ias_dct

    def store_analysis_results(self):
        """This function stores the prediction results and the SHAP values / Shap interaction values as .JSON files."""
        os.makedirs(self.spec_output_path, exist_ok=True)
        print(self.spec_output_path)

        # CV results
        cv_results = self.repeated_nested_scores
        cv_results_filename = os.path.join(
            self.spec_output_path,
            self.var_cfg["analysis"]["output_filenames"]["performance"]
        )
        with open(cv_results_filename, "w") as file:
            json.dump(cv_results, file, indent=4)

        # SHAP values
        shap_values = self.shap_results
        shap_values["feature_names"] = self.X.columns.tolist()
        shap_values_filename = os.path.join(
            self.spec_output_path,
            self.var_cfg["analysis"]["output_filenames"]["shap_values"]
        )
        with open(shap_values_filename, 'wb') as f:
            pickle.dump(shap_values, f)

        # SHAP IA values
        if self.shap_ia_values:
            shap_ia_values = self.shap_ia_values
            shap_ia_values["feature_names"] = self.X.columns
            shap_ia_values_filename = os.path.join(
                self.spec_output_path,
                self.var_cfg["analysis"]["output_filenames"]["shap_ia_values"]
            )
            with open(shap_ia_values_filename, "w") as file:
                json.dump(shap_ia_values, file, indent=4)

        # Lin model coefficients
        if self.lin_model_coefs:
            lin_model_coefs = self.lin_model_coefs
            lin_model_coefs_filename = os.path.join(
                self.spec_output_path,
                self.var_cfg["analysis"]["output_filenames"]["lin_model_coefs"]
            )
            with open(lin_model_coefs_filename, "w") as file:
                json.dump(lin_model_coefs, file, indent=4)

    def get_average_coefficients(self):
        """Gets the coefficients of linear models of the predictions, implemented in the linear model subclass."""
        pass

    def calculate_shap_values(self, X, pipeline):  # TODO Make common method and use explainer as
        """
        This function calculates tree-based SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.TreeExplainer, the SHAP implementation that is
        suitable for tree-based models. SHAP calculations can be parallelized.
        Further, it calculates the SHAP interaction values based on the TreeExplainer, if specified

        Args:
            X: df, features for the machine learning analysis according to the current specification
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: SHAP interaction values, of shape (n_features x n_features x n_samples)
        """
        columns = X.columns
        X_processed = pipeline.named_steps["preprocess"].transform(X)  # Still need this for scaling

        print(self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"])
        if self.model_name == "elasticnet":
            explainer = shap.LinearExplainer(pipeline.named_steps["model"].regressor_, X_processed)
        elif self.model_name == "randomforestregressor":
            explainer = shap.explainers.Tree(pipeline.named_steps["model"].regressor_)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")

        # Compute SHAP values for chunks of the data
        n_jobs = self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"]
        chunk_size = X_processed.shape[0] // n_jobs + (X_processed.shape[0] % n_jobs > 0)
        print("chunk_size:", chunk_size)

        # Compute SHAP values for chunks of the data in parallel
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.calculate_shap_for_chunk)(
                explainer, X_processed[i: i + chunk_size]
            )
            for i in range(0, X_processed.shape[0], chunk_size)
        )

        # Collect and combine results from all chunks
        shap_values_lst, base_values_lst = zip(*results)
        shap_values_array = np.vstack(shap_values_lst)
        base_values_array = np.concatenate(base_values_lst)

        return shap_values_array, base_values_array, columns

    def calculate_shap_for_chunk(self, explainer, X_subset):
        """Calculates tree-based SHAP values for a chunk for parallelization.

        Args:
            explainer: shap.TreeExplainer
            X_subset: df, subset of X for which interaction values are computed, can be parallelized

        Returns:

        """
        # self.logger.log(f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}")
        # Compute SHAP values for the chunk
        explanation = explainer(X_subset)
        shap_values = explanation.values
        base_values = explanation.base_values
        return shap_values, base_values

    @staticmethod
    def spearman_corr(y_true_raw, y_pred_raw):
        """
        For using it as a scorer in the nested cv scheme, we need to manually calculate spearmans rank correlation.

        Args:
            y_true_raw: pd.Series, the "true" reactivities that we predict in the machine learning-based prediction procedure
            y_pred_raw: pd.Series, the predicted reactivities that were outputted by the model

        Returns:
            rank_corr: spearmans rank correlation (rho) for given y_true and y_pred values
        """
        y_true = y_true_raw
        y_pred = y_pred_raw
        rank_corr, _ = spearmanr(y_true, y_pred)
        return rank_corr

    def log_thread(self):
        """
        Logs the current process ID and thread name, along with the method name in which this method is executed.
        """
        process_id = os.getpid()
        thread_name = threading.current_thread().name

        # Use inspect to get the name of the calling function
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)
        method_name = caller_frame[1].function

        print(f"        Executing {method_name} using Process ID: {process_id}, Thread: {thread_name}")
        self.logger.log(f"        Executing {method_name} using Process ID: {process_id}, Thread: {thread_name}")
