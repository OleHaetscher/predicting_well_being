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

    def apply_methods(self):
        """This function applies the preprocessing methods specified in the config."""
        for method in self.var_cfg["postprocessing"]["significance_tests"]["methods"]:
            if method not in dir(SignificanceTesting):
                raise ValueError(f"Method '{method}' is not implemented yet.")
            getattr(self, method)()

    def get_cv_results(self):
        """
        This method extracts all CV results from a file. It loads the JSON files and stores them
        in a dictionary that mirrors the result directory structure

        Returns:

        """

    def conduct_significance_tests(self):
        """
        Wrapper method for all the significance testing. This method
            - computes the significance tests for the model and feature comparisons
            - applies a joint FDR correction to all p-values
            - creates a tabular-like df containing all p and t values



        Returns:

        """
        pass
        # Iterate over the dict that contains the relevant comparisons
        # conduct the comparisons

        # results_compare_models = self.compare_models(dct)
        # results_compare_predictor_classes = self.compare_predictor_classes(dct)

    def compare_models(self, dct):
        """
        This method compares the predictive performance of ENR and RFR across all analysis

        Args:
            dct:

        Returns:
        """
        pass

    def compare_predictor_classes(self, dct):
        """
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
        pass


    def extract_metric(self):
        """
        This method gets the metric used for the comparisons

        Returns:

        """


    def fdr_correct_p_values(self, result_dict):
        """
        Correct p-values using False Discovery Rate (FDR) as described by Benjamini & Hochberg (1995)

        Args:
            result_dict: Dict, containing all results for a certain analysis setting (e.g., main/ssc)

        Returns:
              corrected_p_values_dct: Dict, same structure as result_dict, but with corrected p_values
        """
        p_values = []
        labels = []
        for esm_sample, data in result_dict.items():
            for soc_int_var, comparisons in data.items():
                for model_pair, stats in comparisons.items():
                    p_value = stats["p"]
                    p_values.append(p_value)
                    labels.append((esm_sample, soc_int_var, model_pair))
        adjusted_p_values = fdrcorrection(p_values)[1]
        # format the p_values for the table accordingly
        formatted_p_values = self.format_p_values(adjusted_p_values)
        corrected_p_values_dct = {
            label: formatted_p for label, formatted_p in zip(labels, formatted_p_values)
        }
        return corrected_p_values_dct

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




