from statsmodels.stats.multitest import fdrcorrection

from src.utils.DataLoader import DataLoader

import json
import os
from itertools import combinations
from math import sqrt
from statistics import stdev

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
from statsmodels.stats.multitest import fdrcorrection

from collections import deque, defaultdict


class SignificanceTesting:
    """
    This class computes test of significance to compare the prediction results for different models
    across different analysis settings. Results for different feature selection strategies were pooled.
        Thus, in Study 1 (ssc / main analysis), 6 comparisons (pairwise comparisons of models) are computed
        for each ESM sample - soc_int_var combination, resulting in 42 statistical tests.
        In Study 2, 6 comparisons are computed for each event, resulting in 18 statistical tests.
    Due to multiple testing, tests of significance are False-Discovery-Rate corrected.
    Results are stored as a table for the supplementary results and as a JSON that is used by the CVResultPlotter
    to include the results of the significance tests as annotations in the CV result plots.

    Attributes:
        config: YAML config determining certain specifications of the analysis.
        result_dct: Dict, the predictions results are loaded from its folders and stored in this Dict.
        fis_aggregated_results: Dict,
        significance_results: Dict,
    """

    def __init__(
        self,
            var_cfg
    ):
        """
        Constructor method of the SignificanceTesting Class.

        Args:
            config_path: Path to the .YAML config file.
        """
        self.var_cfg = var_cfg
        self.base_output_dir = self.var_cfg["postprocessing"]["significance_tests"]["output_path"]
        self.result_dct = None
        self.metric = self.var_cfg["postprocessing"]["significance_tests"]["metric"]
        self.data_loader = DataLoader()

        self.compare_model_results = {}
        self.compare_predictor_classes_results = {}

    def significance_testing(self, dct):
        """
        Wrapper function that
            - compares models
            - compares predictor classes
            - applies fdr correction based on both results
            - returns final t and p values

        compare_models
            This method compares the predictive performance of ENR and RFR across all analysis

        compare_predictor_classes
            This method evaluates of adding other predictor classes to person-level predictors leads to
            a significant performance increase.
            Specifically, we compare the addition of
                - srmc
                - sens
                - srmc + sens
                - mac
                - srmc + mac
            using
                - all samples that includes a lot of missings (i.e., all vs. all)
                - selected samples to avoid missings (i.e., selected vs. control)
            seperately for both prediction models
                - ENR
                - RFR
            which results in 20 statistical tests

        Args:
            dct:

        Returns:

        """
        # compare models
        data_for_compare_models = self.get_data(raw_dct=dct, comparison="models")
        self.apply_compare_models(data_for_compare_models)
        data_for_compare_predictor_classes = self.get_data(raw_dct=dct, comparison="predictor_classes")
        self.apply_compare_predictor_classes(data_for_compare_predictor_classes)

        self.fdr_correct_p_values()

    def get_data(self, raw_dct: dict, comparison: str) -> dict:
        """

        Args:
            comparison:

        Returns:
            dict: Dict containing the processed data for significance testing (e.g., the right metric for enr and rfr in
                the final values).

        """
        # Initialize the result dictionary
        processed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # Check if the comparison is based on models
        if comparison == "models":
            # Traverse the raw dictionary
            for fc, fc_vals in raw_dct.items():
                for sti, sti_values in fc_vals.items():
                    # Traverse the models under 'state_wb'
                    if "state_wb" in sti_values:  # remove
                        for model, model_vals in sti_values["state_wb"].items():

                            processed_data[fc][sti][model] = (
                                        self.extract_metric_across_folds_imps(
                                            data_dct=model_vals["cv_results"],
                                            metric=self.metric
                                        )
                                    )

        elif comparison == "predictor_classes":
            # Fill full data
            for fc, fc_vals in raw_dct.items():
                if "all" in fc_vals:  # remove  -> no results yet, skip
                    if "state_wb" in fc_vals["all"]:  # remove
                        for model, models_vals in fc_vals["all"]["state_wb"].items():
                            if "pl" in fc:
                                processed_data["full_data"][model][fc] = (
                                        self.extract_metric_across_folds_imps(
                                            data_dct=models_vals["cv_results"],
                                            metric=self.metric
                                        )
                                    )
            # Fill reduced data
            for fc, fc_vals in raw_dct.items():
                if "control" in fc_vals or "selected" in fc_vals:
                    for sti, sti_vals in fc_vals.items():
                        if "state_wb" in sti_vals:
                            for model, models_vals in sti_vals["state_wb"].items():
                                if "pl_" in fc:
                                    processed_data["reduced_data"][model][fc][sti] = (
                                        self.extract_metric_across_folds_imps(
                                            data_dct=models_vals["cv_results"],
                                            metric=self.metric
                                        )
                                    )
        else:
            raise ValueError
        return processed_data

    @staticmethod
    def extract_metric_across_folds_imps(data_dct, metric: str):
        """
        This function extracts a given metric across outer folds and imputations for the significance tests

        Args:
            data_dct: Dict with the levels reps/outer_folds/imp that contain the metrics as values
            metric: Metric the significance tests are based on (r2)

        Returns:
            metric_values (deque): A deque containing the 500 metric values

        """
        metric_values = deque()
        # Traverse replications, outer folds, and imputations
        for rep_key, rep_vals in data_dct.items():
            for fold_key, fold_vals in rep_vals.items():
                for imp_key, imp_vals in fold_vals.items():
                    # Extract the 'r2' value and add to the deque
                    r2_val = imp_vals.get(metric, None)
                    if r2_val is not None:
                        metric_values.append(r2_val)
        return metric_values

    def apply_compare_models(self, processed_data: dict) -> None:
        """
        This function applies the corrected dependent t-test to compare models in the given processed data.

        Args:
            processed_data (dict): A dictionary structured as fc/sti/models where models contain lists or deques of r2 values.

        Returns:
            dict: The same processed_data dictionary, but with test results (p_val, t_val, etc.) replacing the original lists.
        """

        for fc, fc_vals in processed_data.items():
            for sti, model_data in fc_vals.items():
                if len(model_data) < 2:
                    print(f"WARNING: Not enough models to compare in {fc}/{sti}, SKIP")
                    # raise ValueError(f"Not enough models to compare in {fc}/{sti}")
                    continue

                # Extract model names and their associated r2 value lists (or deque)
                model_names = list(model_data.keys())
                model1_name = model_names[0]
                model2_name = model_names[1]

                data1 = model_data[model1_name]  # Get the r2 values for model1
                data2 = model_data[model2_name]  # Get the r2 values for model2

                # Perform the corrected dependent t-test
                t_val, p_val = self.corrected_dependent_ttest(data1, data2)

                # Replace the r2 values with the test results
                processed_data[fc][sti] = {
                    'model_comparison': f"{model1_name} vs {model2_name}",
                    'p_val': p_val,
                    't_val': np.round(t_val, 2)
                }
        self.compare_model_results = processed_data

    def apply_compare_predictor_classes(self, processed_data: dict) -> None:
        """

        Args:
            processed_data:

        Returns:

        """
        # Iterate over models
        for model, model_vals in processed_data["full_data"].items():
            # First, identify the 'pl' feature class values
            if "pl" not in model_vals:
                print(f"WARNING: 'pl' feature class not found for model {model}, SKIP")
                continue

            pl_vals = model_vals["pl"]  # Get the 'pl' feature class values

            # Now, iterate through the other feature classes and compare them with 'pl'
            for fc, fc_vals in model_vals.items():
                if fc == "pl":
                    continue  # Skip the 'pl' comparison with itself

                # Perform the corrected dependent t-test between 'pl' and the current feature class
                t_val, p_val = self.corrected_dependent_ttest(pl_vals, fc_vals)

                # Store the comparison results for the current feature class
                processed_data["full_data"][model][fc] = {
                    'comparison_with_pl': f"{fc} vs pl",
                    'p_val': p_val,
                    't_val': round(t_val, 2)
                }

        for model, model_vals in processed_data["reduced_data"].items():
            for fc, fc_vals in model_vals.items():
                if len(fc_vals) < 2:
                    print(f"WARNING: Not enough data yet, SKIP")
                    # raise ValueError(f"Not enough models to compare in {fc}/{sti}")
                    continue
                # Perform the corrected dependent t-test
                t_val, p_val = self.corrected_dependent_ttest(fc_vals["selected"], fc_vals["control"])
                processed_data["reduced_data"][model][fc] = {
                'model_comparison': f"{fc} vs pl",
                'p_val': p_val,
                't_val': round(t_val, 2)
                }

        self.compare_predictor_classes_results = processed_data

    def fdr_correct_p_values(self):
        """
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995).
        This function works recursively to find all instances of 'p_val' in a nested dictionary structure.

        Returns:
            result_dict: dict, with corrected 'p_val_corrected' values added to the same structure.
        """
        p_values = []
        p_val_locations = []

        # Helper function to recursively traverse the dictionary
        def find_p_values(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    # Recursively search nested dictionaries
                    find_p_values(value)
                elif key == "p_val":
                    # Collect the p_val and its dictionary reference
                    p_values.append(value)
                    p_val_locations.append((d, key))

        # Start the recursive search
        find_p_values(self.compare_model_results)
        find_p_values(self.compare_predictor_classes_results)
        print()

        # Apply FDR correction on collected p-values
        if p_values:
            adjusted_p_values = fdrcorrection(p_values)[1]

            # Format the p_values for the table accordingly (if you have a format function)
            formatted_p_values = self.format_p_values(adjusted_p_values)

            # Insert the corrected p-values back into the same dictionary locations
            for i, (p_val_dict, key) in enumerate(p_val_locations):
                p_val_dict["p_val_corrected"] = formatted_p_values[i]
        print()

    @staticmethod
    def corrected_dependent_ttest(data1, data2, test_training_ratio=1 / 9):
        """
        Python implementation for the corrected paired t-test as described by Nadeau & Bengio (2003).

        Args:
            data1: list, containing the prediction results for a certain setting (up to a specific model)
            data2: list, containing the prediction results for a another setting (up to a specific model)
            test_training_ratio: float, depends on the number of folds in the outer_cv (i.e., 10 in this setting)

        Returns:
            t_stat: float, t statistic of the comparison of data1 and data2
            p: float, p-value for the comparison of data1 and data2
        """
        n = len(data1)
        differences = [(data1[i] - data2[i]) for i in range(n)]
        sd = stdev(differences)
        divisor = 1 / n * sum(differences)
        denominator = sqrt(1 / n + test_training_ratio) * sd
        t_stat = np.round(divisor / denominator, 2)
        df = n - 1  # degrees of freedom
        p = np.round((1.0 - t.cdf(abs(t_stat), df)) * 2.0, 4)  # p value
        return t_stat, p

    @staticmethod
    def format_p_values(lst_of_p_vals):
        """
        This function formats the p_values according to APA standards (3 decimals, <.001 otherwise)

        Args:
            lst_of_p_vals: list, containing the p_values for a given analysis setting

        Returns:
            formatted_p_vals: list, contains p_values formatted according to APA standards
        """
        formatted_p_vals = []
        for p_val in lst_of_p_vals:
            if p_val < 0.001:
                formatted_p_vals.append("<.001")
            else:
                formatted = "{:.3f}".format(p_val).lstrip("0")
                formatted_p_vals.append(formatted)
        return formatted_p_vals




