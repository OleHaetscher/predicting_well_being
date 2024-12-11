import re

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

from src.utils.utilfuncs import defaultdict_to_dict, format_df


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
        self.sig_cfg = self.var_cfg["postprocessing"]["significance_tests"]
        self.base_result_dir = self.var_cfg["postprocessing"]["significance_tests"]["base_result_path"]
        self.result_dct = None
        self.metric = self.sig_cfg["metric"]
        self.data_loader = DataLoader()

        self.compare_model_results = {}
        self.compare_predictor_classes_results = {}

    def significance_testing(self):
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
        if self.sig_cfg["compare_models"]:
            data_to_compare_models = self.get_model_comparison_data()
            sig_results_models = self.apply_compare_models(data_to_compare_models)
            sig_results_models_fdr = self.fdr_correct_p_values(sig_results_models)
            sig_results_models_table = self.create_sig_results_table_models(sig_results_models_fdr)
            if self.sig_cfg["store"]:
                file_name = os.path.join(self.base_result_dir, "compare_models.xlsx")
                sig_results_models_table.to_excel(file_name, index=True)

        # compare predictor classes
        if self.sig_cfg["compare_predictor_classes"]:
            data_to_compare_predictor_classes = self.get_predictor_class_comparison_data()
            sig_results_predictor_classes = self.apply_compare_predictor_classes(data_to_compare_predictor_classes)
            sig_results_predictor_classes_fdr = self.fdr_correct_p_values(sig_results_predictor_classes)
            sig_results_predictor_classes_table = self.create_sig_results_table_predictor_classes(sig_results_predictor_classes_fdr)
            if self.sig_cfg["store"]:
                file_name = os.path.join(self.base_result_dir, "compare_predictor_classes.xlsx")
                sig_results_predictor_classes_table.to_excel(file_name, index=True)

    def create_sig_results_table_models(self, data_dct):
        """
        Creates a pandas DataFrame from the nested dictionary structure.

        Args:
            data_dct:

        Returns:
            pd.DataFrame: A DataFrame with:
                          - Columns based on the first dictionary hierarchy (`outer_key`).
                          - Multi-index rows:
                              - Level 1: Inner dictionary keys (`second_key`).
                              - Level 2: Metrics (`p_val`, `t_val`, `p_val_corrected`).
        """
        # Flatten the dictionary into rows for easier DataFrame creation
        flattened_data = []

        # Iterate through the nested dictionary structure
        for feature_combo, inner_dict in data_dct.items():
            for samples_to_include, metrics in inner_dict.items():
                for stat, stat_value in metrics.items():
                    # Add each metric as a separate row
                    flattened_data.append({
                        "Predictor class": feature_combo,
                        "Samples to include": samples_to_include,
                        "Stat": stat,
                        "Stat value": stat_value
                    })

        # Create a flat DataFrame
        df = pd.DataFrame(flattened_data)

        # Format the table
        df["Predictor class"] = df["Predictor class"].map(
            self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"]
        )
        # Set custom order for stats
        custom_order = self.sig_cfg["stat_order"]
        df["Stat"] = pd.Categorical(df["Stat"], categories=custom_order, ordered=True)
        df["Stat"] = df["Stat"].map(self.sig_cfg["stat_mapping"])

        # Pivot the DataFrame
        df_pivoted = df.pivot(index=["Samples to include", "Stat"], columns="Predictor class", values="Stat value")

        # Format the pivoted DataFrame
        df_pivoted = format_df(df_pivoted, decimals=2)

        return df_pivoted

    def create_sig_results_table_predictor_classes(self, data_dct):
        """
        Creates a pandas DataFrame from the nested dictionary structure for predictor classes.

        Args:
            data_dct: The nested dictionary structure.

        Returns:
            pd.DataFrame: A DataFrame with:
                          - Columns based on the last dictionary keys before the statistics (e.g., `pl_mac`).
                          - Multi-index rows:
                              - Level 1: Model (e.g., `elasticnet`).
                              - Level 2: Samples to include (e.g., `selected`).
                              - Level 3: Statistics (e.g., `p_val`, `t_val`, `p_val_corrected`).
        """
        # Flatten the dictionary into rows for easier DataFrame creation
        flattened_data = []

        # Iterate through the nested dictionary structure
        for model, sample_dict in data_dct.items():
            for sample, feature_dict in sample_dict.items():
                for feature, metrics in feature_dict.items():
                    for metric_key, metric_value in metrics.items():
                        # Add each metric as a separate row
                        flattened_data.append({
                            "Prediction model": model,
                            "Samples to include": sample,
                            "Predictor class": feature,
                            "Stat": metric_key,
                            "Stat value": metric_value
                        })

        # Create a flat DataFrame
        df = pd.DataFrame(flattened_data)

        # Format the table
        df["Predictor class"] = df["Predictor class"].map(
            self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"]
        )
        # Set custom order for stats
        custom_order = self.sig_cfg["stat_order"]
        df["Stat"] = pd.Categorical(df["Stat"], categories=custom_order, ordered=True)
        df["Stat"] = df["Stat"].map(self.sig_cfg["stat_mapping"])

        # Pivot the DataFrame to create the desired structure
        df_pivoted = df.pivot(index=["Prediction model", "Samples to include", "Stat"], columns="feature", values="Stat value")

        df_pivoted = format_df(df_pivoted, decimals=2)

        return df_pivoted

    def get_model_comparison_data(self):
        """
        This method loads the JSON files containing the CV results data necessary to compare the models.

        Returns:
            A Dict containing the relevant data for testing in the following format:
            {
                "pl": {
                    "all": {
                        "randomforestregressor": [list with 500 values],
                        "elasticnet": [list with 500 values]
                    },
                    "selected": {
                        ...
                    },
                },
                ...
            }
        """

        # Extract configuration
        feature_combos = self.sig_cfg["model_comparison_data"]["feature_combinations"]
        samples_to_include = self.sig_cfg["model_comparison_data"]["samples_to_include"]
        metric = self.sig_cfg["metric"]  # e.g. "r2"














































        models = ["elasticnet", "randomforestregressor"]

        # Precompile pattern
        file_pattern = re.compile(r"cv_results_rep_\d+\.json")

        # Using a nested defaultdict to avoid repeated setdefault calls
        # Structure: results[feature_combo][sample][model] = list of metrics
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for dirpath, dirnames, filenames in os.walk(self.base_result_dir):
            # Normalize and split directory path for checks
            dir_components = os.path.normpath(dirpath).split(os.sep)

            # Check if this directory contains one of the desired feature combos and samples
            feature_combo = next((fc for fc in feature_combos if fc in dir_components), None)
            sample = next((s for s in samples_to_include if s in dir_components), None)
            if not feature_combo or not sample:
                continue

            # Identify the model from the directory path if present
            model = next((m for m in models if m in dir_components), None)
            if not model:
                continue

            # Process JSON files that match the pattern
            for filename in filenames:
                if not file_pattern.match(filename):
                    continue

                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Extract the metric values
                for outer_fold_data in data.values():
                    for metrics in outer_fold_data.values():
                        if metric in metrics:
                            results[feature_combo][sample][model].append(metrics[metric])

        results_dct = defaultdict_to_dict(results)
        return results_dct

    def get_predictor_class_comparison_data(self):
        """
        Loads the JSON files containing the CV results data for comparing models (predictor classes).

        The returned structure:
        {
            model: {
                "all": {
                    feature_combo: [metrics...],
                    ...
                },
                "selected": {
                    feature_combo: [metrics...],
                    ...
                }
            },
            ...
        }

        Logic:
        - "all": take data from the "all" sample directly.
        - "selected":
            * For "pl", take data from "control" sample.
            * For other feature combos, take data from "selected" sample.
        """

        # Extract configuration details
        feature_combos = self.sig_cfg["predictor_class_comparison_data"]["feature_combinations"]
        samples_to_include = self.sig_cfg["predictor_class_comparison_data"]["samples_to_include"]
        metric = self.sig_cfg["metric"]

        file_pattern = re.compile(r"cv_results_rep_\d+\.json")

        # Collect raw data
        # raw_data[model][sample][feature_combo] = list of metric values
        raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for dirpath, _, filenames in os.walk(self.base_result_dir):
            dir_components = os.path.normpath(dirpath).split(os.sep)

            # Identify feature_combo and sample if present
            feature_combo = next((fc for fc in feature_combos if fc in dir_components), None)
            sample = next((s for s in samples_to_include if s in dir_components), None)

            if not feature_combo or not sample:
                continue

            # Identify model:
            # The model should be a directory component that is neither a sample nor a feature combo.
            # We assume model names appear in dir_components.
            # Filter out known samples and feature combos.
            possible_models = [c for c in dir_components if c not in samples_to_include and c not in feature_combos]
            if not possible_models:
                # If we can't find a model in the path, skip
                continue

            # Assume the last remaining component that isn't a sample or feature combo is the model
            # If there are multiple, choose the first. Adjust logic if needed.
            model = possible_models[-1]

            # Process files
            for filename in filenames:
                if not file_pattern.match(filename):
                    continue

                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)

                # Extract the specified metric
                for outer_fold_data in data.values():
                    for metrics_data in outer_fold_data.values():
                        if metric in metrics_data:
                            raw_data[model][sample][feature_combo].append(metrics_data[metric])

        final_results = {}
        for model, model_data in raw_data.items():
            all_pairs = []
            selected_pairs = []

            # Cache the baseline 'pl' data for the 'all' scenario
            all_pl = model_data.get("all", {}).get("pl", [])

            # For the 'selected' scenario, 'pl' is derived from each combo under 'control'.
            # We'll get this inside the loop.

            # For each feature combination (except 'pl'), create pairs with 'pl'
            for c in feature_combos:
                if c == "pl":
                    # Skip 'pl' itself, as we only create pairs comparing 'pl' with another combo
                    continue

                # For 'all':
                # 'pl' from model_data["all"]["pl"]
                # c from model_data["all"][c]
                c_all_data = model_data.get("all", {}).get(c, [])
                all_pairs.append({
                    "pl": all_pl,
                    c: c_all_data
                })

                # For 'selected':
                # 'pl' data comes from c/control
                # c data comes from c/selected
                pl_selected_data = model_data.get("control", {}).get(c, [])
                c_selected_data = model_data.get("selected", {}).get(c, [])
                selected_pairs.append({
                    "pl": pl_selected_data,
                    c: c_selected_data
                })

            final_results[model] = {
                "all": all_pairs,
                "selected": selected_pairs
            }

        return final_results

    def apply_compare_models(self, processed_data: dict) -> None:
        """
        This function applies the corrected dependent t-test to compare models in the given processed data.

        Args:
            processed_data (dict): A dictionary structured as fc/sti/models where models contain lists or deques of r2 values.

        Returns:
            dict: The same processed_data dictionary, but with test results (p_val, t_val, etc.) replacing the original lists.
        """
        sig_results_dct = defaultdict(lambda: defaultdict(dict))
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

                # Get results into the result_dct
                sig_results_dct[fc][sti] = {
                    # 'model_comparison': f"{model1_name} vs {model2_name}",
                    'p_val': p_val,
                    't_val': np.round(t_val, 2)
                }

        sig_results_dct = defaultdict_to_dict(sig_results_dct)

        return sig_results_dct

    def apply_compare_predictor_classes(self, processed_data: dict) -> None:
        """

        Args:
            processed_data:

        Returns:

        """
        # Iterate over models
        sig_results_dct = defaultdict(lambda: defaultdict(dict))
        for model, model_vals in processed_data.items():  # enr / rfr
            for samples_to_include, samples_to_include_vals in model_vals.items():  # all / selected
                for comparison in samples_to_include_vals:  # just numbers: TODO Use second key of child
                    dct_values = list(comparison.values())
                    pl_combo_key = list(comparison.keys())[1]
                    pl_data = dct_values[0]
                    pl_combo_data = dct_values[1]

                    # Perform the corrected dependent t-test between 'pl' and the current feature class
                    if pl_data and pl_combo_data:
                        pl_m, pl_sd = np.mean(pl_data), np.std(pl_data)
                        pl_combo_m, pl_combo_sd = np.mean(pl_combo_data), np.std(pl_combo_data)
                        t_val, p_val = self.corrected_dependent_ttest(pl_data, pl_combo_data)

                        # Get results into the result_dct
                        sig_results_dct[model][samples_to_include][pl_combo_key] = {
                            'p_val': p_val,
                            't_val': np.round(t_val, 2)
                        }

        sig_results_dct = defaultdict_to_dict(sig_results_dct)
        return sig_results_dct

    def fdr_correct_p_values(self, data_dct):
        """
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995).
        This function works recursively to find all instances of 'p_val' in a nested dictionary structure.

        Returns:
            result_dict: dict, with corrected 'p_val_corrected' values added to the same structure.
        """
        p_values = []
        p_val_locations = []

        # Helper function to recursively traverse the dictionary
        def find_p_values(d, path=None):
            if path is None:
                path = []

            for key, value in d.items():
                current_path = path + [key]  # Extend the path with the current key
                if isinstance(value, dict):
                    # Recursively search nested dictionaries
                    find_p_values(value, current_path)
                elif key == "p_val":
                    # Collect the p_val and its dictionary reference
                    p_values.append(value)
                    p_val_locations.append(d)  # Store the dictionary where the p_val exists

        # Start the recursive search
        find_p_values(data_dct)

        # Apply FDR correction on collected p-values
        if p_values:
            adjusted_p_values = fdrcorrection(p_values)[1]

            # Format the p_values for the table accordingly (if you have a format function)
            formatted_p_values_fdr = self.format_p_values(adjusted_p_values)
            # Also format the original p values
            formatted_p_values = self.format_p_values(p_values)

            # Insert the corrected p-values back into the same dictionary locations
            for i, p_val_dct in enumerate(p_val_locations):
                p_val_dct["p_val_fdr"] = formatted_p_values_fdr[i]
                p_val_dct["p_val"] = formatted_p_values[i]

        return data_dct  # Return the updated dictionary

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
            elif p_val < 0.01:
                formatted = "{:.3f}".format(p_val).lstrip("0")
                formatted_p_vals.append(formatted)
            else:
                formatted = "{:.2f}".format(p_val).lstrip("0")
                formatted_p_vals.append(formatted)
        return formatted_p_vals




