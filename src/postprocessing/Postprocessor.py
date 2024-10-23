import copy
import os

import numpy as np

from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.utils.DataLoader import DataLoader

import pandas as pd


class Postprocessor:
    """
    This class executes the different postprocessing steps. This includes
        - conducting tests of significance to compare prediction results
        - calculate and display descriptive statistics
        - creates plots (results, SHAP, SHAP interaction values)
    """

    def __init__(self, fix_cfg, var_cfg, name_mapping):
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.name_mapping = name_mapping
        self.data_loader = DataLoader()
        self.descriptives_creator = DescriptiveStatistics(fix_cfg=self.fix_cfg, var_cfg=self.var_cfg, name_mapping=name_mapping)
        self.significance_testing = SignificanceTesting(var_cfg=self.var_cfg)
        self.shap_processor = ShapProcessor(var_cfg=self.var_cfg)
        self.base_result_dir = self.var_cfg["analysis"]["results_cluster_path"]
        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]

        self.cv_results_dict = {}

    def process_cv_results(self):
        """
        This function
            - iterates over all subdirectories of self.base_result_dir
            - stores the content of the result files "cv_results.json" in a dictionary that mirros the folder structure
            - computes M
            - computes SD
                1) across outer folds and imputations
                2) across outer folds within imputations
                3) across imputations within outer folds
            - prepares the data for significance-testing
        Returns:
        """
        result_dct = self.get_cv_results()
        result_dct_processed = self.compute_m_sd_cv_results(result_dct)
        # self.plot_results(result_dct_processed)
        print()

    def get_cv_results(self) -> dict:
        """
        # TODO: Make this more general, than apply for "cv_results" and "lin_model_coefs"
        This function:
            - Iterates over all subdirectories of self.base_result_dir
            - Stores the content of the result files "cv_results.json" in a dictionary that mirrors the folder structure
        Returns:
            dict: Nested dict containing the cv_results of all analysis in self.base_result_dir
        """
        results_dict = {}  # Initialize the dictionary to store results

        for root, dirs, files in os.walk(self.base_result_dir):
            cv_results_filename = self.result_filenames["performance"]
            if cv_results_filename in files:
                # Build the relative path from base_result_dir to cv_results.json's directory
                rel_path = os.path.relpath(root, self.base_result_dir)
                # Split the relative path into parts
                path_parts = rel_path.split(os.sep)
                # The path_parts list will be used to build the nested dictionary

                # Load the result_file
                cv_results_path = os.path.join(root, 'cv_results.json')
                cv_results = self.data_loader.read_json(cv_results_path)

                # Build the nested dictionary structure
                current_level = results_dict
                for part in path_parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                # At this point, current_level is the innermost dictionary corresponding to the path
                # Store the cv_results here
                current_level['cv_results'] = cv_results

        return results_dict

    def compute_m_sd_cv_results(self, result_dict: dict) -> dict:
        """
        This function computes statistics of the results obtained, it
            - computes the mean (m) across all 500 outer folds
            - computes the standard deviation (sd)
                1) across outer folds and imputations
                2) across outer folds within imputations
                3) across imputations within outer folds

        Args:
            result_dict (dict): Nested dict containing the result of all analysis in self.base_result_dir

        Returns:
            dict: Nested dict containing the m and sd calculated
        """
        # Create a deep copy to avoid modifying the original dictionary
        result_dict_copy = copy.deepcopy(result_dict)

        # Recursive function to traverse and process the dictionary
        def process_node(node):
            for key in list(node.keys()):
                if key == 'cv_results':
                    # Process the 'cv_results' at this level
                    cv_results = node[key]
                    data_records = []

                    # Collect all metrics into a list of records
                    for rep_key, rep_value in cv_results.items():
                        for fold_key, fold_value in rep_value.items():
                            for imp_key, imp_value in fold_value.items():
                                # imp_value is a dict of metrics
                                record = {
                                    'rep': int(rep_key.split('_')[-1]),
                                    'outer_fold': int(fold_key.split('_')[-1]),
                                    'imputation': int(imp_key.split('_')[-1])
                                }
                                record.update(imp_value)
                                data_records.append(record)

                    # Convert to DataFrame
                    df = pd.DataFrame(data_records)
                    # Identify metric columns
                    identifier_cols = ['rep', 'outer_fold', 'imputation']
                    metric_cols = [col for col in df.columns if col not in identifier_cols]

                    # Compute mean across all data points (m)
                    group_mean = df[metric_cols].mean().round(3).to_dict()

                    # Compute standard deviation across all data points (sd_across_folds_imps)
                    sd_across_folds_imps = df[metric_cols].std(ddof=0).round(4).to_dict()

                    # Compute standard deviation across outer folds within imputations (sd_within_folds_across_imps)
                    sd_within_folds_across_imps = df.groupby('imputation')[metric_cols].std(ddof=0).mean().round(4).to_dict()

                    # Compute standard deviation across imputations within outer folds (sd_across_folds_within_imps)
                    sd_across_folds_within_imps = df.groupby('outer_fold')[metric_cols].std(ddof=0).mean().round(4).to_dict()

                    # Replace 'cv_results' with computed statistics
                    node['m'] = group_mean
                    node['sd_across_folds_imps'] = sd_across_folds_imps
                    node['sd_within_folds_across_imps'] = sd_within_folds_across_imps
                    node['sd_across_folds_within_imps'] = sd_across_folds_within_imps

                    # Remove the raw 'cv_results' data
                    del node['cv_results']

                elif isinstance(node[key], dict):
                    # Recursively process the sub-dictionary
                    process_node(node[key])

        # Start processing from the root of the copied dictionary
        process_node(result_dict_copy)

        return result_dict_copy

    def plot_results(self, result_stats):
        """
        This function creates plots of the results.
        - For each criterion (e.g., 'state_wb')
        - For each condition (e.g., 'control')
        - In one big plot, it plots all different feature combinations (e.g., 'pl_srmc_mac')
        - In this big plot, it plots the mean and sd across outer folds and imputations for the 'r2' metric
        - For both 'elasticnet' and 'random forest', in subplots
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect all conditions and criteria from the result_stats
        conditions = set()
        criteria = set()
        models = set()
        for fc_dict in result_stats.values():
            for condition, cond_dict in fc_dict.items():
                conditions.add(condition)
                for criterion, crit_dict in cond_dict.items():
                    criteria.add(criterion)
                    models.update(crit_dict.keys())

        models = list(models)

        # Iterate over each condition and criterion
        for condition in conditions:
            for criterion in criteria:
                # Initialize data containers for plotting
                feature_combinations = []
                data_per_model = {model: {'means': [], 'sds': []} for model in models}

                # Collect data for each feature combination
                for feature_combination, fc_dict in result_stats.items():
                    if condition in fc_dict and criterion in fc_dict[condition]:
                        crit_dict = fc_dict[condition][criterion]
                        for model in models:
                            if model in crit_dict:
                                stats = crit_dict[model]
                                mean_r2 = stats['m']['r2']
                                sd_r2 = stats['sd_across_folds_imps']['r2']
                                data_per_model[model]['means'].append(mean_r2)
                                data_per_model[model]['sds'].append(sd_r2)
                        feature_combinations.append(feature_combination)

                if not feature_combinations:
                    continue  # Skip if no data is available for this condition and criterion

                num_models = len(models)
                fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharey=True)
                if num_models == 1:
                    axes = [axes]

                x = np.arange(len(feature_combinations))
                for i, model in enumerate(models):
                    ax = axes[i]
                    means = data_per_model[model]['means']
                    sds = data_per_model[model]['sds']
                    if means:
                        ax.bar(x, means, yerr=sds, align='center', alpha=0.7, ecolor='black', capsize=5)
                        ax.set_xticks(x)
                        ax.set_xticklabels(feature_combinations, rotation=45, ha='right')
                        ax.set_ylabel('R2 Score')
                        ax.set_title(f'{model}')
                        ax.yaxis.grid(True)
                    else:
                        ax.set_visible(False)

                fig.suptitle(f'Criterion: {criterion} - Condition: {condition}', fontsize=16)
                # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
                plt.show()



    def postprocess(self):
        # self.process_cv_results()
        self.shap_processor.recreate_explanation_objects()
        self.descriptives_creator.create_m_sd_feature_table()
        self.descriptives_creator.create_wb_item_statistics()
        self.significance_testing.apply_methods()
