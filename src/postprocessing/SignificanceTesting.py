import copy
import re
from typing import Any

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
                file_name = os.path.join(self.base_result_dir, "sig_table_compare_models.xlsx")
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

    def create_sig_results_table_models(self, p_val_dct: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from the nested dictionary structure.

        Args:
            p_val_dct:

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
        for feature_combo, inner_dict in p_val_dct.items():
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

        # Set custom order for stats
        custom_order_stats = self.sig_cfg["stat_order_compare_models"]
        df["Stat"] = pd.Categorical(df["Stat"], categories=custom_order_stats, ordered=True)
        df["Stat"] = df["Stat"].map(self.sig_cfg["stat_mapping"])

        # Set custom order for predictor classes
        custom_order_predictor_classes = self.sig_cfg["model_comparison_data"]["feature_combinations"]
        df["Predictor class"] = pd.Categorical(df["Predictor class"], categories=custom_order_predictor_classes, ordered=True)
        df["Predictor class"] = df["Predictor class"].map(
            self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"]
        )

        # Pivot the DataFrame
        df_pivoted = df.pivot(index=["Samples to include", "Stat"], columns="Predictor class", values="Stat value")
        return df_pivoted

    def create_sig_results_table_predictor_classes(self, p_val_dct: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]) -> pd.DataFrame:
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
        for model, sample_dict in p_val_dct.items():
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

        # Right column order (set later)
        col_order = [val for key, val in self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"].items()
                     if key in df["Predictor class"].unique()]

        # Format the table
        df["Predictor class"] = df["Predictor class"].map(
            self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"]
        )

        # Set 'Stat' column as a categorical with the ordered categories
        custom_order = self.sig_cfg["stat_order_compare_predictor_classes"]
        df["Stat"] = pd.Categorical(df["Stat"], categories=custom_order, ordered=True)
        df["Stat"] = df["Stat"].apply(lambda x: self.sig_cfg["stat_mapping"].get(x, x))

        # Pivot the DataFrame to create the desired structure
        df_pivoted = df.pivot(index=["Prediction model", "Samples to include", "Stat"], columns="Predictor class", values="Stat value")

        df_pivoted = df_pivoted[col_order]
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

            if "wb_state" not in dir_components:
                continue

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
        Loads and processes cross-validation (CV) results data from JSON files to generate a structured comparison
        of performance metrics across predictor classes (models) and feature combinations.

        This method traverses the base results directory and identifies relevant CV result files based on directory
        structure and file naming patterns. It extracts the performance metrics for specific feature combinations
        and sample types ("all", "selected", and "control") and organizes them into a structured format for
        comparing models.

        The resulting structure (`final_results`) returned by this method is as follows:
        {
            model_name: {
                "all": [
                    {
                        "pl": [...],   # Baseline metrics for "pl" under "all" sample
                        feature_combo: [...]  # Metrics for the current feature combination under "all"
                    },
                    ...
                ],
                "selected": [
                    {
                        "pl": [...],   # Baseline metrics derived from "control" data
                        feature_combo: [...]  # Metrics for the current feature combination under "selected"
                    },
                    ...
                ]
            },
            ...
        }

        Logic Overview:
        - For the "all" scenario:
          * Metrics for "pl" are taken from "all"/"pl" data.
          * Metrics for other feature combinations are paired with "pl" under "all".
        - For the "selected" scenario:
          * Metrics for "pl" are derived from the "control" data for each feature combination.
          * Metrics for other feature combinations are taken from the "selected" data.
        - The method processes all CV result files matching the `cv_results_rep_\d+\.json` pattern within
          directories that include a known feature combination and sample type.
        - Performance metrics are aggregated based on the specified `metric_to_use` from the configuration.

        Steps Performed:
        1. Traverse the result directory to identify valid paths containing the required feature combinations and samples.
        2. Parse JSON files matching the file naming pattern and extract relevant performance metrics.
        3. Organize the data into a hierarchical structure based on models, sample types, and feature combinations.
        4. Generate metric pairs for comparison between the "pl" baseline and other feature combinations under
           both "all" and "selected" scenarios.

        Returns:
            dict: A structured dictionary containing comparison data for each model, with results categorized
                  under "all" and "selected" scenarios.
        """
        feature_combos = self.sig_cfg["predictor_class_comparison_data"]["feature_combinations"]
        samples_to_include = self.sig_cfg["predictor_class_comparison_data"]["samples_to_include"]
        metric_to_use = self.sig_cfg["metric"]
        file_pattern = re.compile(r"cv_results_rep_\d+\.json")
        raw_data = defaultdict(lambda: {
            "all": defaultdict(list),
            "selected": defaultdict(list),
            "control": defaultdict(list)
        })

        for dirpath, _, filenames in os.walk(self.base_result_dir):
            dir_components = os.path.normpath(dirpath).split(os.sep)
            if "wb_state" not in dir_components:
                continue

            feature_combo = next((fc for fc in feature_combos if fc in dir_components), None)
            sample = next((s for s in samples_to_include if s in dir_components), None)
            if feature_combo is None or sample is None:
                continue
            possible_models = [c for c in dir_components if c not in samples_to_include and c not in feature_combos]
            if not possible_models:
                continue
            model = possible_models[-1]

            if sample == "all":
                target_key = "all"
            elif sample == "control":
                target_key = "control"
            elif sample == "selected":
                target_key = "selected"
            else:
                continue

            for filename in filenames:
                if not file_pattern.match(filename):
                    continue
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                for _, outer_fold_data in data.items():
                    for imp_data in outer_fold_data.values():
                        if metric_to_use in imp_data:
                            raw_data[model][target_key][feature_combo].append(imp_data[metric_to_use])

        final_results = {}
        for model, model_data in raw_data.items():
            all_pairs = []
            selected_pairs = []
            all_pl = model_data["all"]["pl"]

            for c in feature_combos:
                if c == "pl":
                    continue

                c_all_data = model_data["all"][c]
                all_pairs.append({
                    "pl": all_pl,
                    c: c_all_data
                })
                pl_selected_data = model_data["control"][c]
                c_selected_data = model_data["selected"][c]
                selected_pairs.append({
                    "pl": pl_selected_data,
                    c: c_selected_data
                })

            final_results[model] = {
                "all": all_pairs,
                "selected": selected_pairs
            }

        return final_results

    def apply_compare_models(self, cv_results_dct: dict[str, dict[str, dict[str, list[float]]]]) -> dict[str, dict[str, dict[str, float]]]:
        """
        This function applies the corrected dependent t-test to compare models in the given processed data. It
            - iterates through the nested Dictionary
            - extracts the values obtained from 10x10x10 CV for self.metric
            - conducts the corrected dependent t-test between the two models
            - stores the results in a dict that mirrors the original structure

        Args:
            cv_results_dct: Nested dict that contains the CV results for self.metric as a list in the inner dict structure
                and the outer dict hierarchy contains the feature combinations to include, the samples to include (all/selected),
                and the model to include (elasticnet/randomforestregressor).

        Returns:
            dict: The same processed_data dictionary, but with test results (p_val, t_val, etc.) replacing the original lists.
        """
        sig_results_dct = defaultdict(lambda: defaultdict(dict))
        for fc, fc_vals in cv_results_dct.items():
            for sti, model_data in fc_vals.items():
                if len(model_data) < 2:
                    print(f"WARNING: Not enough models to compare in {fc}/{sti}, SKIP")
                    continue

                model_names = list(model_data.keys())

                model1_name = model_names[0]
                model2_name = model_names[1]

                cv_results_model1 = model_data[model1_name]
                cv_results_model2 = model_data[model2_name]

                cv_results_model1_mean, cv_results_model1_sd = np.mean(cv_results_model1), np.std(cv_results_model1)
                cv_results_model2_mean, cv_results_model2_sd = np.mean(cv_results_model2), np.std(cv_results_model2)

                t_val, p_val = self.corrected_dependent_ttest(cv_results_model1, cv_results_model2)

                sig_results_dct[fc][sti] = {
                    f"{model1_name} M": np.round(cv_results_model1_mean, 3),
                    f"{model1_name} SD": np.round(cv_results_model1_sd, 3),
                    f"{model2_name} M": np.round(cv_results_model2_mean, 3),
                    f"{model2_name} SD": np.round(cv_results_model2_sd, 3),
                    f"deltaR2": np.round(cv_results_model1_mean - cv_results_model2_mean, 3),
                    'p_val': p_val,
                    't_val': np.round(t_val, 2)
                }

        sig_results_dct = defaultdict_to_dict(sig_results_dct)

        return sig_results_dct

    def apply_compare_predictor_classes(self, cv_results_dct: dict[
        str, dict[str, list[dict[str, list[float]]]]]) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
        """

        Args:
            processed_data:

        Returns:

        """
        # Iterate over models
        sig_results_dct = defaultdict(lambda: defaultdict(dict))
        for model, model_vals in cv_results_dct.items():  # enr / rfr
            for samples_to_include, samples_to_include_vals in model_vals.items():  # all / selected
                for comparison in samples_to_include_vals:
                    dct_values = list(comparison.values())
                    pl_combo_key = list(comparison.keys())[1]
                    pl_data = dct_values[0]
                    pl_combo_data = dct_values[1]

                    # Perform the corrected dependent t-test between 'pl' and the current feature class
                    if pl_data and pl_combo_data:
                        cv_results_pl_m, cv_results_pl_sd = np.mean(pl_data), np.std(pl_data)
                        cv_results_pl_combo_m, cv_results_pl_combo_sd = np.mean(pl_combo_data), np.std(pl_combo_data)
                        t_val, p_val = self.corrected_dependent_ttest(pl_data, pl_combo_data)

                        # pl_combo_key_formatted = self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"][pl_combo_key]

                        # Get results into the result_dct
                        sig_results_dct[model][samples_to_include][pl_combo_key] = {
                            f"MR2 (Personal)": np.round(cv_results_pl_m, 3),
                            f"SDR2 (Personal)": np.round(cv_results_pl_sd, 3),
                            f"MR2 (Personal + Other)": np.round(cv_results_pl_combo_m, 3),
                            f"SDR2 (Personal + Other)": np.round(cv_results_pl_combo_sd, 3),
                            f"deltaR2": np.round(cv_results_pl_combo_m - cv_results_pl_m, 3),
                            'p_val': p_val,
                            't_val': np.round(t_val, 2)
                        }

        sig_results_dct = defaultdict_to_dict(sig_results_dct)
        return sig_results_dct

    def fdr_correct_p_values(self, p_val_dct: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
        """ # TODO: Does the signature apply to both comparisons?
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995).
        This function
            - recursively finds all instances of 'p_val' in a nested dictionary structure
            - apply the FDR correction
            - formats the p-values if a formatting function is provided
            - recreates the dict structure of the input and adds the FDR corrected p-values

        Args:
            p_val_dct: Dict containing the t-values and the uncorrected p-values for each comparison to conduct.
                The outer dict hierarchy depends on the type of comparison (models vs. predictor classes).

        Returns:
            result_dict: Dict, with 'p_val_fdr' values added in the inner Dict.
        """
        p_val_fdr_dct = copy.deepcopy(p_val_dct)
        p_values = []
        p_val_locations = []

        def collect_p_values(d: dict[str, Any]) -> None:
            """
            Recursively collect all 'p_val' values and their containing dictionaries.

            Args:
                d: The Dict to traverse.
            """
            for key, value in d.items():
                if isinstance(value, dict):
                    collect_p_values(value)
                elif key == "p_val":
                    p_values.append(value)
                    p_val_locations.append(d)

        collect_p_values(p_val_fdr_dct)

        if p_values:
            _, adjusted_p_values = fdrcorrection(p_values)

            if hasattr(self, 'format_p_values') and callable(getattr(self, 'format_p_values')):
                formatted_p_values_fdr = self.format_p_values(adjusted_p_values)
                formatted_p_values = self.format_p_values(p_values)
            else:
                formatted_p_values_fdr = adjusted_p_values
                formatted_p_values = p_values

            for loc, adj_p, orig_p in zip(p_val_locations, formatted_p_values_fdr, formatted_p_values):
                loc["p_val_fdr"] = adj_p
                loc["p_val"] = orig_p

        return p_val_fdr_dct

    @staticmethod
    def corrected_dependent_ttest(data1: list[float], data2: list[float], test_training_ratio: float = 1/9):
        """
        Python implementation for the corrected paired t-test as described by Nadeau & Bengio (2003).
        See also https://gist.github.com/jensdebruijn/13e8eeda85eb8644ac2a4ac4c3b8e732

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
        print(df)
        p = np.round((1.0 - t.cdf(abs(t_stat), df)) * 2.0, 4)  # p value

        return t_stat, p

    @staticmethod
    def format_p_values(lst_of_p_vals: list[float]) -> list[str]:
        """
        This function formats the p_values according to APA standards (3 decimals, <.001 otherwise)

        Args:
            lst_of_p_vals: list, containing the p_values for a given analysis setting.

        Returns:
            formatted_p_vals: list, contains p_values formatted according to APA style.
        """
        formatted_p_vals = []

        for p_val in lst_of_p_vals:
            if p_val < 0.001:
                formatted_p_vals.append("<.001")
            else:
                formatted = "{:.3f}".format(p_val).lstrip("0")
                formatted_p_vals.append(formatted)

        return formatted_p_vals




