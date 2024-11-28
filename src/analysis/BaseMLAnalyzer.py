import gc
import hashlib
import inspect
import itertools
import json
import os
import pickle
import sys
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import shap
import sklearn
from joblib import Parallel, delayed, Memory, parallel_backend
from scipy.stats import spearmanr, pearsonr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    make_scorer,
    get_scorer, accuracy_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.analysis.CustomScaler import CustomScaler
from src.analysis.Imputer import Imputer
from src.analysis.PearsonFeaureSelector import PearsonFeatureSelector
from src.analysis.ShuffledGroupKFold import ShuffledGroupKFold
from src.utils.DataLoader import DataLoader
from src.utils.Logger import Logger
from src.utils.Timer import Timer
# from mpi4py import MPI  # TODO REMOVE THIS

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("future.no_silent_downcasting", True)


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
        rep,
        rank=None
    ):
        """
        Constructor method of the BaseMLAnalyzer Class.

        Args:
            var_cfg: YAML var_cfg determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        self.var_cfg = var_cfg
        self.spec_output_path = spec_output_path

        # Multi-node parallelism
        self.rank = rank
        # Joblib backend
        self.joblib_backend = self.var_cfg["analysis"]["parallelize"]["joblib_backend"]
        self.rep = rep
        self.split_reps = self.var_cfg["analysis"]["split_reps"]

        self.crit = self.var_cfg["analysis"]["params"]["crit"]
        self.feature_combination = self.var_cfg["analysis"]["params"]["feature_combination"]
        self.samples_to_include = self.var_cfg["analysis"]["params"]["samples_to_include"]

        self.logger = Logger(log_dir=self.spec_output_path, log_file=self.var_cfg["general"]["log_name"], rank=self.rank, rep=self.rep)
        self.timer = Timer(self.logger)
        self.data_loader = DataLoader()

        # Methods that get clocked
        self.repeated_nested_cv = self.timer._decorator(self.repeated_nested_cv)
        self.nested_cv = self.timer._decorator(self.nested_cv)
        self.summarize_shap_values_outer_cv = self.timer._decorator(self.summarize_shap_values_outer_cv)
        self.manual_grid_search = self.timer._decorator(self.manual_grid_search)
        self.impute_datasets = self.timer._decorator(self.impute_datasets)
        self.calculate_shap_ia_values = self.timer._decorator(self.calculate_shap_ia_values)

        # Data
        self.df = df
        self.X = None
        self.y = None
        self.rows_dropped_crit_na = None

        # Results
        self.best_models = {}
        self.repeated_nested_scores = {}
        self.pred_vs_true = {}
        self.shap_results = {'shap_values': {}, "base_values": {}, "data": {}}
        self.shap_ia_results = {'shap_ia_values': {}, "base_values": {}}
        self.shap_ia_results_reps_imps = {'shap_ia_values': {}, "base_values": {}}  # Store these only on the cluster due to big size

        self.shap_ia_results_processed = {'top_interactions': {},  # this can be transfered to the local computer
                                          'top_interacting_features': {},
                                          'ia_value_agg_reps_imps_samples': {},
                                          'ia_values_sample': {},
                                          'base_values_sample': {},
                                          }
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
        self.country_grouping_col = self.var_cfg["analysis"]["imputation"]["country_grouping_col"]
        self.years_col = self.var_cfg["analysis"]["imputation"]["years_col"]

        # Attributes for shap_ia_values
        self.combo_index_mapping = None
        self.feature_index_mapping = None
        self.num_combos = None

        self.meta_vars = [self.id_grouping_col, self.country_grouping_col, self.years_col]

        # Output filenames (without type endings)
        self.performance_name = self.var_cfg["analysis"]["output_filenames"]["performance"]  # .json
        self.shap_value_name = self.var_cfg["analysis"]["output_filenames"]["shap_values"]  # .pkl
        self.shap_ia_values_for_local_name = self.var_cfg["analysis"]["output_filenames"]["shap_ia_values_for_local"]  # .pkl
        self.shap_ia_values_for_cluster_name = self.var_cfg["analysis"]["output_filenames"]["shap_ia_values_for_cluster"] # .pkl
        self.lin_model_coefs_name = self.var_cfg["analysis"]["output_filenames"]["lin_model_coefs"] # .json

    @property
    def model_name(self):
        """Get a string repr of the model name and sets it as class attribute (e.g., "lasso")."""
        return self.model.__class__.__name__.lower()

    @property
    def imputer(self):
        return Imputer(
            logger=self.logger,
            model=self.model_name,
            fix_rs=self.var_cfg["analysis"]["random_state"],
            max_iter=self.var_cfg["analysis"]["imputation"]["max_iter"],
            num_imputations=self.var_cfg["analysis"]["imputation"]["num_imputations"],
            conv_thresh=self.var_cfg["analysis"]["imputation"]["conv_thresh"],
            tree_max_depth=self.var_cfg["analysis"]["imputation"]["tree_max_depth"],
            percentage_of_features=self.var_cfg["analysis"]["imputation"]["percentage_of_features"],
            n_features_thresh=self.var_cfg["analysis"]["imputation"]["n_features_thresh"],
            sample_posterior=self.var_cfg["analysis"]["imputation"]["sample_posterior"],
            pmm_k=self.var_cfg["analysis"]["imputation"]["pmm_k"],
            country_group_by=self.country_grouping_col,
            years_col=self.years_col,
        )

    @property
    def hyperparameter_grid(self):
        """Set hyperparameter grid defined in var_cfg for the specified model as class attribute."""
        return self.var_cfg["analysis"]["model_hyperparameters"][self.model_name]

    def apply_methods(self, comm=None):
        """This function applies the analysis methods specified in the var_cfg."""
        for method_name in self.var_cfg["analysis"]["methods_to_apply"]:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' is not implemented yet.")
            # If split_reps is True, skip 'get_average_coefficients' and 'process_all_shap_ia_values'
            if self.split_reps and method_name in ['get_average_coefficients', 'process_all_shap_ia_values']:
                log_message = f"Skipping {method_name} because split_reps is True."
                self.logger.log(log_message)
                print(log_message)
                continue  # Skip this method
            log_message = f"  Executing {method_name}"
            self.logger.log(log_message)
            print(log_message)
            # Dynamically call the method
            method_to_call = getattr(self, method_name)
            # If the method is 'repeated_nested_cv', pass comm; otherwise, call without comm
            if method_name == 'repeated_nested_cv':
                method_to_call(comm)
            else:
                method_to_call()

    def select_samples(self):
        """
        This method selects the samples based on the given combination using the indices that correspond
        to the samples (e.g., cocoesm_1). It applies the following logic:
            - for the analysis "selected" and "control", only selected datasets are used to reduce NaNs
            - for the analysis "all", all datasets are used, independent of the features used for the analysis
            - For the supplementary analyses: If a dataset does not contain the criterion, it is excluded
        """
        if self.samples_to_include in ["selected", "control"]:
            datasets_included = self.var_cfg["analysis"]["feature_sample_combinations"][self.feature_combination]
            datasets_included_filtered = [dataset for dataset in datasets_included if dataset in
                                          self.var_cfg["analysis"]["crit_available"][self.crit]]
            self.datasets_included = datasets_included_filtered
            df_filtered = self.df[self.df.index.to_series().apply(
                lambda x: any(x.startswith(sample) for sample in self.datasets_included))]

            # for "sens" and "selected", remove samples without sensing data
            if self.samples_to_include in ["selected", "control"] and "sens" in self.feature_combination:
                sens_columns = [col for col in self.df.columns if col.startswith("sens_")]
                # df_filtered = self.df[self.df[sens_columns].notna().any(axis=1)]   # This was wrong
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

            # New -> Add control analysis with reduced samples
            if "_control" in self.feature_combination:
                sens_columns = [col for col in self.df.columns if col.startswith("sens_")]
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

        else:  # include all datasets with available criterion
            datasets_included_filtered = [dataset for dataset in self.var_cfg["analysis"]["feature_sample_combinations"]["all_in"]
                                          if dataset in self.var_cfg["analysis"]["crit_available"][self.crit]]
            self.datasets_included = datasets_included_filtered
            df_filtered = self.df[self.df.index.to_series().apply(
                lambda x: any(x.startswith(sample) for sample in self.datasets_included))]

        # It may also be possible, that some people have NaNs on the trait wb measures, exclude them
        crit_col = f"crit_{self.crit}"
        df_filtered_crit_na = df_filtered.dropna(subset=[crit_col])
        self.rows_dropped_crit_na = len(df_filtered) - len(df_filtered_crit_na)

        # Sample for testing, if defined in the config
        if self.var_cfg["analysis"]["tests"]["sample"]:
            sample_size = self.var_cfg["analysis"]["tests"]["sample_size"]
            df_filtered_crit_na = df_filtered_crit_na.sample(n=sample_size,
                                                             random_state=self.var_cfg["analysis"]["random_state"])

        self.df = df_filtered_crit_na

    def select_features(self):
        """
        This method loads the machine learning features (traits) according to the specifications in the var_cfg.
            - Some columns are always added that are used for the imputation process or data splitting
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "X".
        ADJUST!!!!
        """
        selected_columns = [self.id_grouping_col, self.country_grouping_col, self.years_col]
        if self.samples_to_include in ["all", "selected"]:
            feature_prefix_lst = self.feature_combination.split("_")

            if self.feature_combination == "all_in":  # include all features
                feature_prefix_lst = ["pl", "srmc", "sens", "mac"]
                if self.samples_to_include == "selected":
                    self.logger.log(f"    WARNING: No selected analysis needed for {self.feature_combination}, stop computations")
                    sys.exit(0)

        elif self.samples_to_include == "control":
            feature_prefix_lst = ["pl"]

            no_control_lst = self.var_cfg["analysis"]["no_control_lst"]
            if self.feature_combination in no_control_lst:
                self.logger.log(f"    WARNING: No control analysis needed for {self.feature_combination}, stop computations")
                sys.exit(0)
        else:
            raise ValueError(f"Invalid value #{self.samples_to_include}# for attr samples_to_include")

        if self.feature_combination == "all":  # include all features
            feature_prefix_lst = ["pl", "srmc", "sens", "mac"]

        # always include grouping id
        for feature_cat in feature_prefix_lst:
            for col in self.df.columns:
                if col.startswith(feature_cat):
                    selected_columns.append(col)

        # remove neuroticism facets and self-esteem for selected analysis
        print(len(selected_columns))
        if "nnse" in self.feature_combination:
            to_remove = ["pl_depression", "pl_anxiety", "pl_emotional_volatility", "pl_self_esteem"]
            selected_columns = [col for col in selected_columns if col not in to_remove]
        print(len(selected_columns))
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

        self.logger.log("Parallelization params")
        self.logger.log(f'    Use mpi4py to parallelize across nodes: {self.var_cfg["analysis"]["use_mpi4py"]}')
        self.logger.log(f'    Split reps: {self.split_reps}')
        self.logger.log(f'    Current rank: {self.rank}')
        self.logger.log(f'    N jobs inner_cv: {self.var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"]}')
        self.logger.log(f'    N jobs shap: {self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"]}')
        self.logger.log(f'    N jobs imputation runs: {self.var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"]}')
        self.logger.log(f'    Compute shap IA values: {self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]}')
        if self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]:
            self.logger.log(f'    N jobs shap_ia_values: {self.var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"]}')

        self.logger.log("Data params")
        self.logger.log(f"    Number of rows: {self.X.shape[0]}")
        self.logger.log(f"    Number of cols: {self.X.shape[1]}")
        self.logger.log(f"    Number of rows dropped due to missing criterion {self.crit}: {self.rows_dropped_crit_na}")
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
            if unique_values <= 1:  # drop columns with only NaN or zero variance
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
        preprocessor = CustomScaler()
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

        # create pipeline for feature selection
        if "_fs" in self.feature_combination:
            self.logger.log("    -> Include feature selection for sensing features")
            feature_selector = PearsonFeatureSelector(
                num_features=self.var_cfg["analysis"]["feature_selection"]["num_sensing_features"],
                target_prefix="sens_")
            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("feature_selection", feature_selector),
                    ("model", model_wrapped),
                ],
                memory=memory
            )
        else:
            pipe = Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", model_wrapped),
                ],
                memory=memory
            )
        setattr(self, "pipeline", pipe)

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
        pred_vs_true_rep = {}
        X_train_imputed_lst = []
        X_test_imputed_lst = []

        # Loop over outer cross-validation splits
        for cv_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y, groups=X[self.id_grouping_col])):
            self.logger.log("-----------------------------------")
            self.logger.log("-----------------------------------")
            self.logger.log("#########")
            self.logger.log(f"Outer fold number {cv_idx}, going from 0 to {self.num_outer_cv - 1}")
            self.logger.log("#########")

            ml_pipelines_sublst = []
            ml_model_params_sublst = []
            nested_scores_rep[f"outer_fold_{cv_idx}"] = {}
            pred_vs_true_rep[f"outer_fold_{cv_idx}"] = {}

            # Convert indices and select data
            train_indices = X.index[train_index]
            test_indices = X.index[test_index]
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]

            # Ensure indices match between X and y and group splitting works
            assert (X_train.index == y_train.index).all(), "Indices between train data differ"
            assert (X_test.index == y_test.index).all(), "Indices between test data differ"
            assert len(set(X_train[self.id_grouping_col]).intersection(set(X_test[self.id_grouping_col]))) == 0, \
                "Grouping in outer_cv did not work as expected"

            # Impute datasets
            print(f"now imputing datasets for fold {cv_idx}")
            imputed_datasets = self.impute_datasets(X_train_outer_cv=X_train,
                                                    X_test_outer_cv=X_test,
                                                    y_train_outer_cv=y_train,
                                                    inner_cv=inner_cv,
                                                    num_imputations=self.num_imputations,
                                                    groups=X_train[self.id_grouping_col],
                                                    n_jobs=self.var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"]
                                                    )
            # Store imputed datasets of outer folds for SHAP value calculations
            X_train_imputed_lst_for_fold = [
                imputed_datasets[num_imp]["outer_fold"]["X_train_for_test_full"]
                for num_imp in imputed_datasets
            ]
            X_train_imputed_lst.append(X_train_imputed_lst_for_fold)
            X_test_imputed_lst_for_fold = [
                imputed_datasets[num_imp]["outer_fold"]["X_test_imputed"]
                for num_imp in imputed_datasets
            ]
            X_test_imputed_lst.append(X_test_imputed_lst_for_fold)

            # Get scorers and grid
            scoring_inner_cv = get_scorer(self.var_cfg["analysis"]["scoring_metric"]["inner_cv_loop"]["name"])
            param_grid = list(ParameterGrid(self.hyperparameter_grid))

            for num_imp in range(self.num_imputations):
                nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{num_imp}"] = {}
                grid_search_results = self.manual_grid_search(
                    imputed_datasets=imputed_datasets[num_imp],
                    y=y_train,
                    param_grid=param_grid,
                    scorer=scoring_inner_cv,
                    inner_cv=inner_cv,
                    n_jobs=self.var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"],
                )
                best_model = grid_search_results['best_model']
                # best_params = grid_search_results['best_params']

                # Append model (elasticnet) or model params (RFR)
                if self.model_name == "elasticnet":
                    ml_model_params_sublst.append(best_model.named_steps["model"].regressor_)
                else:
                    ml_model_params_sublst.append(best_model.named_steps["model"].regressor_.get_params())
                ml_pipelines_sublst.append(best_model)

                # Evaluate the model on the imputed test set and store the metrics
                X_test_current = imputed_datasets[num_imp]["outer_fold"]['X_test_imputed']
                X_test_current = X_test_current.drop(columns=self.meta_vars + [self.id_grouping_col], errors='ignore')
                y_test_current = y_test.loc[X_test_current.index]

                scores = self.get_scores(
                    grid_search=SimpleNamespace(best_estimator_=best_model),
                    X_test=X_test_current,
                    y_test=y_test_current
                )
                for metric, score in scores.items():
                    nested_scores_rep[f"outer_fold_{cv_idx}"][f"imp_{num_imp}"][metric] = score

                if self.var_cfg["analysis"]["store_pred_and_true"]:
                    pred_vs_true_rep[f"outer_fold_{cv_idx}"][f"imp_{num_imp}"] = self.get_pred_and_true_crit(
                        grid_search=SimpleNamespace(best_estimator_=best_model),
                        X_test=X_test_current,
                        y_test=y_test_current
                    )

                # Free up memory
                del best_model
                del X_test_current

            ml_model_params.append(ml_model_params_sublst)
            ml_pipelines.append(ml_pipelines_sublst)

        X_filtered = X.drop(columns=self.meta_vars, errors="ignore")
        # Summarize SHAP values and return all results
        (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,
            ia_base_values,
        ) = self.summarize_shap_values_outer_cv(
            X_train_imputed_lst=X_train_imputed_lst,
            X_test_imputed_lst=X_test_imputed_lst,
            X=X_filtered,
            y=y,
            groups=X[self.id_grouping_col],
            pipelines=ml_pipelines,
            outer_cv=outer_cv
        )

        del ml_pipelines
        gc.collect()

        return (
            nested_scores_rep,
            pred_vs_true_rep,
            ml_model_params,
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,  # mean across imps
            ia_base_values,  # mean across imps
        )

    def manual_grid_search(self,
                           imputed_datasets: dict,
                           y: pd.Series,
                           inner_cv: ShuffledGroupKFold,
                           param_grid: list,
                           scorer: Callable,
                           n_jobs: int):
        """
        Mimics the behavior of GridSearchCV using pre-imputed datasets and allows parallelization across param_grid.
        """
        # Function to evaluate a single parameter combination
        def evaluate_param_combination(params, imputed_datasets, y, inner_cv, scorer, pipeline, meta_vars):
            param_results = []
            for fold in range(inner_cv.get_n_splits()):
                fold_name = f"inner_fold_{fold}"
                dataset = imputed_datasets[fold_name]
                X_full = dataset['X_train_for_val_full']
                y_full = y.loc[X_full.index]

                train_indices = dataset['train_indices']
                val_indices = dataset['val_indices']
                # TODO Remove this, if once tested -> this would be repeated for all param combos
                self.logger.log(f"first three val indices for fold {fold}: {val_indices[:3]}")

                # Prepare data
                X_train = X_full.loc[train_indices].drop(columns=meta_vars, errors='ignore')
                y_train = y_full.loc[train_indices]
                X_val = X_full.loc[val_indices].drop(columns=meta_vars, errors='ignore')
                y_val = y_full.loc[val_indices]

                model = clone(pipeline)
                model.set_params(**params)

                model.fit(X_train, y_train)
                score = scorer(model, X_val, y_val)

                param_results.append(score)

            avg_score = np.mean(param_results)
            return {'params': params, 'score': avg_score}

        # Use joblib's Parallel to evaluate param_grid in parallel
        results = Parallel(
            n_jobs=n_jobs,
            backend=self.joblib_backend,
            verbose=3,
        )(
            delayed(evaluate_param_combination)(
                params,
                imputed_datasets,
                y,
                inner_cv,
                scorer,
                self.pipeline,
                self.meta_vars,
                # self.id_grouping_col
            ) for params in param_grid
        )

        # Find best score and parameters
        best_result = max(results, key=lambda x: x['score'])
        best_score = best_result['score']
        best_params = best_result['params']

        # Now fit the model on the entire training data using best_params
        # Retrieve the 'outer_fold' data
        outer_fold_data = imputed_datasets['outer_fold']
        X_train_full = outer_fold_data['X_train_for_test_full']
        y_train_full = y.loc[X_train_full.index]

        # Prepare data
        X_train_full = X_train_full.drop(columns=self.meta_vars + [self.id_grouping_col], errors='ignore')

        # Fit the model
        best_model = clone(self.pipeline)
        best_model.set_params(**best_params)
        best_model.fit(X_train_full, y_train_full)

        nested_scores_outer_fold_imp = {
            'all_results': results,
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model
        }

        return nested_scores_outer_fold_imp

    def impute_datasets(self,
                        X_train_outer_cv: pd.DataFrame,
                        X_test_outer_cv: pd.DataFrame,
                        y_train_outer_cv: pd.Series,
                        inner_cv: ShuffledGroupKFold,
                        groups: pd.Series,
                        num_imputations: int,
                        n_jobs: int):
        """
        This function imputes the data in one outer_fold for all imputations (5) for the inner_cv (10 train / val splits)
        and the outer cv (refit the imputer on the full training data and transform the test data).
        To use all available CPUs, we parallelize the imputations of single datasets.
        We store the datasets to impute in a dict (for the outer_cv, we use fold_0 as a dummy)

        For the inner_folds, we simply recreate the features before splitting, as the same splits will be executed
        by the GridSearchCV.
        For the outer_fold, we return only X_test (as we will not use X_train anymore, but only the inner_cv specific datasets)

        Args:
            X_train_outer_cv: pd.DataFrame,
            X_test_outer_cv: pd.DataFrame,
            y_train_outer_cv: pd.Series,
            inner_cv: ShuffledGroupKFold,
            groups: pd.Series, grouping column that separates the same individuals
            num_imputations: int,
            n_jobs: int,

        Returns:
            dict[dict[pd.DataFrame]: Dict containing the imputed dataset(s) for the trainig or the test data
        """
        data_dct = {}

        # Inner CV data
        # This is the last time we need the groups, as we store the indices resulting from the splits
        for fold, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_outer_cv, y_train_outer_cv, groups=groups)):
            # Convert indices and select data
            train_indices = X_train_outer_cv.index[train_idx]
            val_indices = X_train_outer_cv.index[val_idx]
            X_train_inner_cv, X_val = X_train_outer_cv.loc[train_indices], X_train_outer_cv.loc[val_indices]
            data_dct[f"inner_fold_{fold}"] = [X_train_inner_cv, X_val, train_indices, val_indices]

        # Outer CV data
        X_train_outer_cv = X_train_outer_cv.drop(self.id_grouping_col, axis=1, errors="ignore")
        X_test_outer_cv = X_test_outer_cv.drop(self.id_grouping_col, axis=1, errors="ignore")
        data_dct["outer_fold"] = [X_train_outer_cv, X_test_outer_cv]

        # Define tasks for parallel processing
        tasks = []
        result_dct = {}
        for num_imp in range(num_imputations):
            result_dct[num_imp] = {}
            for fold, (X_train, X_val_or_test, *indices) in data_dct.items():  # X_train_copy, X_val_or_test_copy,
                result_dct[num_imp][fold] = {}
                tasks.append((fold, num_imp, X_train, X_val_or_test, indices))  # X_train_copy, X_val_or_test_copy,

        # Parallel processing
        # print(n_jobs)
        # print(self.joblib_backend)
        results = Parallel(
            n_jobs=n_jobs,
            backend=self.joblib_backend,
            verbose=3
        )(
            delayed(self.impute_single_dataset)
            (fold, num_imp, X_train, X_val_or_test)
            for fold, num_imp, X_train, X_val_or_test, _ in tasks
        )

        # Fill result_dct with results
        for task_idx, (fold, num_imp, _, _, indices) in enumerate(tasks):
            # Extract the corresponding result from the parallel results
            X_train_result, X_val_or_test_result = results[task_idx]

            if fold.startswith("inner"):
                # Concatenate and reorder using the original X_train_outer_cv index
                recreated_df = pd.concat([X_train_result, X_val_or_test_result])
                recreated_df = recreated_df.reindex(X_train_outer_cv.index)
                train_indices, val_indices = indices

                # Store the train_indices and val_indices
                result_dct[num_imp][fold] = {
                    'X_train_for_val_full': recreated_df,
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }
            else:
                # For "outer", store the results with keys
                result_dct[num_imp][fold] = {
                    'X_train_for_test_full': X_train_result,
                    'X_test_imputed': X_val_or_test_result
                }

        return result_dct

    def impute_single_dataset(self, fold, num_imp, X_train, X_val_or_test):
        """
        Impute a single dataset based on fold and imputation number.

        Args:
            fold: The fold identifier, may be "inner_fold_x", or "outer_fold"
            num_imp: The specific imputation iteration.
            X_train: Training dataset.
            X_val_or_test: Validation dataset or Test dataset, depending on "fold"

        Returns:
            tuple(pd.DataFrame): X_train and X_val / X_test imputed

        """
        # Clone the imputer, log indices
        imputer = clone(self.imputer)
        self.logger.log("--------------------------------------------------")
        self.logger.log(f"    Starting to impute dataset {fold} imputation number {num_imp}")
        self.logger.log(f"      first val indices are {X_val_or_test.index[:3]}")
        self.log_thread()

        # Fit the imputer on the training data and transform training and test data
        imputer.fit(X=X_train, num_imputation=num_imp)  # I need num_imp for different random seeds
        self.logger.log(f"    Number of Cols with NaNs in X_train: {len(X_train.columns[X_train.isna().any()])}")
        X_train_imputed = imputer.transform(X=X_train, num_imputation=num_imp)
        self.logger.log(f"    Number of Cols with NaNs in X_test: {len(X_train.columns[X_val_or_test.isna().any()])}")
        X_val_test_imputed = imputer.transform(X=X_val_or_test, num_imputation=num_imp)

        # Remove meta cols if present in df
        X_train_imputed = X_train_imputed.drop(columns=self.meta_vars, errors="ignore")
        X_val_test_imputed = X_val_test_imputed.drop(columns=self.meta_vars, errors="ignore")

        assert self.check_for_nan(X_train_imputed, dataset_name="X_train_imputed"), "There are NaN values in X_train_imputed!"
        assert self.check_for_nan(X_val_test_imputed, dataset_name="X_test_imputed"), "There are NaN values in X_test_imputed!"

        return X_train_imputed, X_val_test_imputed

    def check_for_nan(self, df, dataset_name=""):
        """
        Function to check for NaN values in the DataFrame.
        Outputs the column name and the number of NaN values if any are found.
        """
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            for col in nan_columns:
                n_nans = df[col].isna().sum()
                self.logger.log(f"{dataset_name} - Column '{col}' has {n_nans} NaN values.")
                print(f"{dataset_name} - Column '{col}' has {n_nans} NaN values.")
        return not bool(nan_columns)

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
            "spearman": make_scorer(self.spearman_corr),
            "pearson": make_scorer(self.pearson_corr)
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

    def get_pred_and_true_crit(self, grid_search, X_test, y_test):
        """
        This method generates a dictionary with sample indices as keys and tuples of predicted
        and true criterion values as values. It ensures that any scaling or transformations
        applied in the pipeline are handled appropriately.

        Args:
            grid_search: The trained grid search object with the best estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            pred_true_dict: A dictionary where keys are sample indices, and values are tuples
                            (predicted_value, true_value).
        """
        # Ensure the best estimator is fitted
        best_model = grid_search.best_estimator_

        # Predict using the best model
        y_pred = best_model.predict(X_test)

        # Create the dictionary with sample indices
        pred_true_dict = {
            idx: (pred, true)
            for idx, pred, true in zip(X_test.index, y_pred, y_test)
        }

        return pred_true_dict

    def compute_shap_for_fold(
        self,
        num_cv_,
        pipeline,
        num_test_indices,
        X_train,
        X_test,
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
        X_train_scaled = pipeline.named_steps["preprocess"].transform(X_train)  # scaler
        X_test_scaled = pipeline.named_steps["preprocess"].transform(X_test)  # scaler

        if "feature_selection" in pipeline.named_steps:  # feature sizes must match, conditionally apply fs
            X_test_scaled = pipeline.named_steps["feature_selection"].transform(X_test)
            original_feature_names = X_test.columns  # Feature names before selection
            selected_feature_names = X_test_scaled.columns if isinstance(X_test_scaled, pd.DataFrame) else pipeline.named_steps[
                "feature_selection"].selected_features_
            # Find the intersection and get the indices in the original dataset
            feature_indices = [original_feature_names.get_loc(name) for name in selected_feature_names if
                               name in original_feature_names]
            for i in feature_indices:  # for testing
                print(i)
        else:
            feature_indices = None

        (
            shap_values_test,
            base_values_test,
        ) = self.calculate_shap_values(X_train_scaled=X_train_scaled,
                                       X_test_scaled=X_test_scaled,
                                       pipeline=pipeline)

        if (
            self.model_name == "randomforestregressor"
            and self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]
        ):
            if self.combo_index_mapping is None:
                combo_index_mapping, feature_index_mapping, num_combos = self.create_index_mappings(X_test)
                self.combo_index_mapping = combo_index_mapping
                self.feature_index_mapping = feature_index_mapping
                self.num_combos = num_combos
            shap_ia_values_arr, shap_ia_base_values_arr = self.calculate_shap_ia_values(
                X_scaled=X_test_scaled,
                pipeline=pipeline,
                combo_index_mapping=self.combo_index_mapping,
            )

        else:
            shap_ia_values_arr = None
            shap_ia_base_values_arr = None

        return (
            shap_values_test,
            base_values_test,
            X_test_scaled,
            # X_test,
            num_test_indices[num_cv_],
            # num_cv_,
            shap_ia_values_arr,
            shap_ia_base_values_arr,
            feature_indices,
        )

    def calculate_shap_ia_values(self, X_scaled: pd.DataFrame, pipeline: Pipeline, combo_index_mapping: dict):
        """
        Placeholder method to compute SHAP interaction values.

        Args:
            X_scaled:
            pipeline:
            combo_index_mapping:

        Returns:
            dict:
            list(float):
        """
        raise NotImplementedError("Subclasses should implement this method")

    def summarize_shap_values_outer_cv(self,
                                       X_train_imputed_lst: list[list[pd.DataFrame]],
                                       X_test_imputed_lst: list[list[pd.DataFrame]],
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       groups: pd.Series,
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

        # Get numerical indices of the samples in the outer cv
        num_test_indices = [test for train, test in outer_cv.split(X, y, groups=groups)]

        # Set up the array to store the results -> 3D arrays (rows x features x imputations), also for feature selection
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
                    num_test_indices=num_test_indices,
                    X_train=X_train_imputed_lst[num_cv_][num_imputation],  # TODO
                    X_test=X_test_imputed_lst[num_cv_][num_imputation],
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
                    X_test_scaled,
                    test_idx,
                    _,
                    _,
                    feature_indices
            ) in enumerate(fold_results):
                # TODO: Test
                # Aggregate results for the current fold and imputation
                if "feature_selection" in pipelines[num_cv_][num_imputation].named_steps:  # we have only a feature subset
                    print(np.shape(rep_shap_values))
                    print(np.shape(shap_values_template))
                    print(len(feature_indices))
                    rep_shap_values[np.ix_(test_idx, feature_indices, [num_imputation])] += shap_values_template.astype(np.float32)[..., np.newaxis]
                    rep_base_values[test_idx, num_imputation] += base_values_template.flatten().astype(np.float32)
                    print(np.shape(rep_data))
                    print(np.shape(X_test_scaled))
                    rep_data[np.ix_(test_idx, feature_indices, [num_imputation])] += X_test_scaled.astype(np.float32).to_numpy()[..., np.newaxis]
                else:  # we have all features
                    rep_shap_values[test_idx, :, num_imputation] += shap_values_template.astype(np.float32)
                    rep_base_values[test_idx, num_imputation] += base_values_template.flatten().astype(np.float32)
                    rep_data[test_idx, :, num_imputation] += X_test_scaled.astype(np.float32)

        # We need to divide the base values, because we get base_values in every outer fold
        rep_base_values = rep_base_values / self.num_outer_cv

        if (
            self.model_name == "randomforestregressor"
            and self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]
        ):
            ia_test_shap_values = np.zeros((X.shape[0], self.num_combos, self.num_imputations), dtype=np.float32)
            ia_test_base_values = np.zeros((X.shape[0], self.num_imputations), dtype=np.float32)  # we get different base values per fold

            # Aggregate results from all folds and imputations for SHAP-IA Values
            for num_cv_, fold_results in enumerate(results):
                for num_imputation, (
                        _,
                        _,
                        _,
                        test_idx,
                        shap_ia_values_arr,
                        shap_ia_base_values_arr,
                        _
                ) in enumerate(fold_results):  # TODO Adding feature selection here may be way more complex.. leave out
                    ia_test_shap_values[test_idx, :, num_imputation] += shap_ia_values_arr
                    ia_test_base_values[test_idx, num_imputation] += shap_ia_base_values_arr

            # For reducing
            ia_test_shap_values = ia_test_shap_values.mean(axis=2)  # Shape: (test_idx, features)
            ia_test_base_values = ia_test_base_values.mean(axis=1)  # Shape: (test_idx)
        else:
            ia_test_shap_values = None
            ia_test_base_values = None

        return (
            rep_shap_values,
            rep_base_values,
            rep_data,
            ia_test_shap_values,  # already mean across imps
            ia_test_base_values,  # already mean across imps
        )

    def create_index_mappings(self, X):
        """
        This function creates a unambiguous mapping between feature combinations (based on their indices)
        and numerical indices in a numpy array, used to store the shap_ia_values efficiently.
        Additionally, it creates a mapping between feature_indices and feature_names, as the SHAP-IQ explainer just return
        feature indices

        Args:
            feature_indices:

        Returns:
            dict:
            dict:
            int:
        """
        min_order_shap = self.var_cfg["analysis"]["shap_ia_values"]["min_order"]
        max_order_shap = self.var_cfg["analysis"]["shap_ia_values"]["max_order"]
        num_features = X.shape[1]
        feature_indices = range(num_features)
        feature_names = X.columns

        feature_to_index = {feature_idx: feature for feature_idx, feature in zip(feature_indices, feature_names)}

        combinations = []
        for r in range(min_order_shap, max_order_shap+1):  # TODO verify that this indeed matches the output
            combinations.extend(itertools.combinations(feature_indices, r))

        # Map each combination to a unique index
        combination_to_index = {idx: combination for idx, combination in enumerate(combinations)}

        return combination_to_index, feature_to_index, len(combinations)

    def repeated_nested_cv(self, comm=None):
        if comm:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            rank = 0
            size = 1

        print(f"Process {rank} of {size} is running")
        self.logger.log(f"Process {rank} of {size} is running")

        X = self.X.copy()
        y = self.y.copy()
        fix_random_state = self.var_cfg["analysis"]["random_state"]

        # Determine repetitions to run
        if comm:
            all_reps = list(range(self.num_reps))
            my_reps = all_reps[rank::size]
            print(f"    Rank {rank}: Handling repetitions {my_reps}")
            self.logger.log(f"    [Rank {rank}] Handling repetitions {my_reps}")
        else:
            # If not using MPI, run only the specified repetition or all if rep is None
            if self.rep is not None:
                my_reps = [self.rep]
            else:
                my_reps = list(range(self.num_reps))
            print(f"    Running repetitions {my_reps}")
            self.logger.log(f"    Running repetitions {my_reps}")

        results = []
        results_file_paths = []
        for rep in my_reps:
            print(f"    Rank {rank}: Starting repetition {rep}")
            self.logger.log(f"    [Rank {rank}] Starting repetition {rep}")

            # Check if data is equal
            data_hash = hashlib.md5(self.X.to_csv().encode()).hexdigest()
            print(f"    Rank {rank} data hash: {data_hash}")
            self.logger.log(f"    Rank {rank} data hash: {data_hash}")

            dynamic_rs = fix_random_state + rep
            nested_scores_rep, pred_vs_true_rep, ml_model_params, rep_shap_values, rep_base_values, rep_data,\
                rep_ia_values, rep_ia_base_values = self.nested_cv(
                rep=rep,
                X=X,
                y=y,
                fix_rs=fix_random_state,
                dynamic_rs=dynamic_rs,
            )

            # Unpack result and exclude large arrays
            result_without_large_arrays = (
                nested_scores_rep,
                pred_vs_true_rep,
                ml_model_params,
                rep_shap_values,
                rep_base_values,
                rep_data,
            )

            results.append((rep, result_without_large_arrays))
            shap_val_dct = {"shap_values": rep_shap_values,
                            "base_values": rep_base_values,
                            "data": rep_data,
                            "feature_names": self.X.columns.tolist()
                            }

            print(f"    [Rank {rank}] Finished repetition {rep}")
            self.logger.log(f"    Rank {rank}: Finished repetition {rep}")

            # Save IA values to files -> Save always due to file size
            if self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]:
                file_name_ia_values = os.path.join(  # TODO Test all this
                    self.spec_output_path, f"shap_ia_values_rep_{rep}.pkl"  # Can I leave out the rank??? If not using mpi4py indeed
                )
                file_name_ia_base_values = os.path.join(
                    self.spec_output_path, f"shap_ia_base_values_rep_{rep}.pkl"
                )
                with open(file_name_ia_values, "wb") as f:
                    pickle.dump(rep_ia_values, f)
                with open(file_name_ia_base_values, "wb") as f:
                    pickle.dump(rep_ia_base_values, f)

                # Store file paths with their corresponding rep number
                results_file_paths.append((rep, file_name_ia_values, file_name_ia_base_values))

                # Store also the mappings for the postprocessing
                ia_values_mappings = {
                    "combo_index_mapping": self.combo_index_mapping.copy(),
                    "feature_index_mapping": self.feature_index_mapping.copy(),
                    "num_combos": self.num_combos.copy()
                }
                file_name_ia_values_mappings = os.path.join(
                    self.spec_output_path, f"ia_values_mappings_rep_{rep}.pkl"
                )
                with open(file_name_ia_values_mappings, "wb") as f:
                    pickle.dump(ia_values_mappings, f)

            # Save models to file
            best_models_file = os.path.join(self.spec_output_path, f"best_models_rep_{rep}.pkl")
            with open(best_models_file, "wb") as f:
                pickle.dump(ml_model_params, f)

            if self.split_reps:  # Store results for single reps, if we use different jobs for different reps

                nested_scores_file = os.path.join(self.spec_output_path, f"cv_results_rep_{rep}.json")
                with open(nested_scores_file, "w") as file:
                    json.dump(nested_scores_rep, file, indent=4)

                pred_vs_true_file = os.path.join(self.spec_output_path, f"pred_vs_true_rep_{rep}.json")
                with open(pred_vs_true_file, "w") as file:
                    json.dump(pred_vs_true_rep, file, indent=4)

                shap_values_file = os.path.join(self.spec_output_path, f"shap_values_rep_{rep}.pkl")
                with open(shap_values_file, "wb") as f:
                    pickle.dump(shap_val_dct, f)

        if comm:
            # MPI case
            all_results = comm.gather((results, results_file_paths), root=0)
            if rank == 0:
                # Collect all results into final_results and all_file_paths
                final_results = []
                all_file_paths = []
                for res, paths in all_results:
                    final_results.extend(res)
                    all_file_paths.extend(paths)
                print(f"  [Rank {rank}] Collected all results")
                self.logger.log(f"  [Rank {rank}] Collected all results")
        else:
            # Non-MPI case
            # Assign results and results_file_paths directly
            final_results = results
            all_file_paths = results_file_paths

        # Process results (only on rank 0 for MPI case)
        if not comm or (comm and rank == 0):
            # Process the final results
            for rep, (
                    nested_scores_rep,
                    pred_vs_true_rep,
                    best_models,
                    rep_shap_values,
                    rep_base_values,
                    rep_data,
            ) in final_results:
                # If we do not split across reps, we keep it as before, even when not using mpi4py
                print(f"Processing rep {rep}")
                self.best_models[f"rep_{rep}"] = best_models
                self.shap_results["shap_values"][f"rep_{rep}"] = rep_shap_values
                self.shap_results["base_values"][f"rep_{rep}"] = rep_base_values
                self.shap_results["data"][f"rep_{rep}"] = rep_data
                self.repeated_nested_scores[f"rep_{rep}"] = nested_scores_rep
                self.pred_vs_true[f"rep_{rep}"] = pred_vs_true_rep

            # Load the large arrays from the file paths
            if self.var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"]:
                for rep, ia_values_path, ia_base_values_path in all_file_paths:
                    with open(ia_values_path, "rb") as f:
                        rep_shap_ia_values_test = pickle.load(f)
                    with open(ia_base_values_path, "rb") as f:
                        rep_shap_ia_base_values = pickle.load(f)

                    self.shap_ia_results["shap_ia_values"][f"rep_{rep}"] = rep_shap_ia_values_test
                    self.shap_ia_results["base_values"][f"rep_{rep}"] = rep_shap_ia_base_values

            try:
                for rep in range(len(self.repeated_nested_scores)):
                    print(f"scores for rep {rep}: ", self.repeated_nested_scores[f"rep_{rep}"])
            except Exception as e:
                print("I do not contain all repetitions")
                self.logger.log("I do not contain all repetitions")

    def store_analysis_results(self):
        """This function stores the prediction results and the SHAP values / SHAP interaction values."""
        if self.rank == 0 and not self.split_reps:
            os.makedirs(self.spec_output_path, exist_ok=True)
            print(self.spec_output_path)

            # Determine the current repetition
            rep = self.rep if self.rep is not None else "all"

            # CV results
            cv_results_filename = os.path.join(
                self.spec_output_path,
                f"{self.performance_name}_rep_{rep}.json"
            )
            with open(cv_results_filename, "w") as file:
                json.dump(self.repeated_nested_scores, file, indent=4)

            # Pred vs true
            pred_vs_true_filename = os.path.join(
                self.spec_output_path,
                f"pred_vs_true_rep_{rep}.json"
            )
            with open(pred_vs_true_filename, "w") as file:
                json.dump(xx, file, indent=4)  # TODO:

            # SHAP values
            self.shap_results["feature_names"] = self.X.columns.tolist()
            shap_values_filename = os.path.join(
                self.spec_output_path,
                f"{self.shap_value_name}_rep_{rep}.pkl"
            )
            with open(shap_values_filename, 'wb') as f:
                pickle.dump(self.shap_results, f)

            # Linear model coefficients
            if self.lin_model_coefs:
                lin_model_coefs_filename = os.path.join(
                    self.spec_output_path,
                    f"{self.lin_model_coefs_name}_rep_{rep}.json"
                )
                with open(lin_model_coefs_filename, "w") as file:
                    json.dump(self.lin_model_coefs, file, indent=4)

    def get_average_coefficients(self):
        """Gets the coefficients of linear models of the predictions, implemented in the linear model subclass."""
        pass

    def process_all_shap_ia_values(self):
        """Aggregates shap interaction values, implemented in the RFR subclass."""
        pass

    def calculate_shap_values(self, X_train_scaled, X_test_scaled, pipeline):
        """
        This function calculates tree-based SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling).
        It calculates the SHAP values using the explainers.TreeExplainer, the SHAP implementation that is
        suitable for tree-based models. SHAP calculations can be parallelized.
        Further, it calculates the SHAP interaction values based on the TreeExplainer, if specified

        Args:
            X_train_scaled: Scaöed train dataset, we need this as background data for the linear shap analyzer
            X_test_scaled: Scaled test dataset where we compute the SHAP values for
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: SHAP interaction values, of shape (n_features x n_features x n_samples)
        """
        # columns = X.columns
        # X_processed = pipeline.named_steps["preprocess"].transform(X)  # Still need this for scaling

        print(self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"])
        if self.model_name == "elasticnet":  # TODO Test this -> Check runtime
            n_samples = len(X_train_scaled)
            explainer = shap.LinearExplainer(model=pipeline.named_steps["model"].regressor_,
                                             masker=X_train_scaled,
                                             n_samples=n_samples)
        elif self.model_name == "randomforestregressor":
            explainer = shap.explainers.Tree(pipeline.named_steps["model"].regressor_)
        else:
            raise ValueError(f"Model {self.model_name} not implemented")

        # Compute SHAP values for chunks of the data
        n_jobs = self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"]
        chunk_size = X_test_scaled.shape[0] // n_jobs + (X_test_scaled.shape[0] % n_jobs > 0)
        print("chunk_size:", chunk_size)

        # Compute SHAP values for chunks of the data in parallel
        results = Parallel(
            n_jobs=n_jobs,
            backend=self.joblib_backend,
            verbose=3,
        )(
            delayed(self.calculate_shap_for_chunk)(
                explainer, X_test_scaled[i: i + chunk_size]
            )
            for i in range(0, X_test_scaled.shape[0], chunk_size)
        )

        # Collect and combine results from all chunks
        shap_values_lst, base_values_lst = zip(*results)
        shap_values_array = np.vstack(shap_values_lst)
        base_values_array = np.concatenate(base_values_lst)

        return shap_values_array, base_values_array

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

    @staticmethod
    def pearson_corr(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]  # Return only the correlation coefficient

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
