import json
from collections import defaultdict
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.utils.DataLoader import DataLoader
from src.utils.Logger import Logger


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
        self.base_result_dir = self.var_cfg["postprocessing"]["raw_results_path"]
        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]
        self.processed_output_path = self.var_cfg["postprocessing"]["processed_results_path"]
        self.cv_results_dct = {}
        self.metrics = self.var_cfg["postprocessing"]["metrics"]
        self.methods_to_apply = self.var_cfg["postprocessing"]["methods"]

        self.data_loader = DataLoader()
        self.logger = Logger(
            log_dir=self.var_cfg["general"]["log_dir"],
            log_file=self.var_cfg["general"]["log_name"]
        )
        self.descriptives_creator = DescriptiveStatistics(
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            name_mapping=name_mapping
        )
        self.significance_testing = SignificanceTesting(
            var_cfg=self.var_cfg
        )
        self.plotter = ResultPlotter(
            var_cfg=self.var_cfg,
            plot_base_dir=self.processed_output_path
        )
        self.shap_processor = ShapProcessor(
            var_cfg=self.var_cfg,
            base_result_dir=self.base_result_dir,
            processed_output_path=self.processed_output_path,
            name_mapping=self.name_mapping,
        )

    def postprocess(self):
        """
        This is kind of a wrapper method that does all the postprocessing steps specified in the config.
        It may invoke the methods in the other classes. What it does:

            sanity_check_pred_vs_true:
                Make post-hoc analyses based on the predicted and the true criterion values
            condense_cv_results:
                Creates tables containing all results for different metrics
            condense_lin_model_coefs:
                Creates tables containing the ENR coeffiecients for all analyses
            create_descriptives:
                Calculates descriptive statistics
                    - M/SD of features, wb_items, and criteria
                    - ICCs and bp/wp correlation of wb_items
                    - reliability of criteria per sample
            conduct_significance_tests:
                Conducts significance tests (corrected paired t-tests) and apply FDR correction
            create_cv_results_plot:
                Creates a plot that visualizes the prediction results for both models and all feature combinations
            create_shap_plots:
                Creates plots visualizing SHAP and SHAP interaction values. This includes
                    SHAP beeswarm plots representing the most imortant features for all feature combinations
                    SHAP importance plots TBA
        """
        if "sanity_check_pred_vs_true" in self.methods_to_apply:
            self.sanity_check_pred_vs_true()

        if "condense_cv_results" in self.methods_to_apply:
            for metric in self.metrics:
                metric_dict, data_points = self.extract_metrics(
                    base_dir=self.processed_output_path,
                    metric=metric,
                    cv_results_filename=self.var_cfg["postprocessing"]["summarized_file_names"]["cv_results"]
                )
                self.create_df_table(
                    data=data_points,
                    metric=metric,
                    output_dir=self.processed_output_path,
                    custom_order=self.var_cfg["postprocessing"]["cv_results"]["table_feature_combo_order"]
                )
                self.cv_results_dct[metric] = metric_dict

        if "condense_lin_model_coefs" in self.methods_to_apply:
            coefficients_dict, coefficient_points = self.extract_coefficients(
                base_dir=self.processed_output_path,
                coef_filename=self.var_cfg["postprocessing"]["summarized_file_names"]["lin_model_coefs"]
            )
            self.create_coefficients_dataframe(
                data=coefficient_points,
                output_dir=self.processed_output_path
            )

        if "create_descriptives" in self.methods_to_apply:
            # pass  TODO Bugfix
            # rel = self.descriptives_creator.compute_rel()
            self.descriptives_creator.create_m_sd_feature_table()
            self.descriptives_creator.create_wb_item_statistics()

        if "conduct_significance_tests" in self.methods_to_apply:
            # TODO: Complete, with simulated data?
            self.significance_testing.significance_testing(dct=self.cv_results_dct.copy())

        if "create_cv_results_plots" in self.methods_to_apply:
            # Create cv_results_plot for all metrics for the supplement
            if self.cv_results_dct:
                for metric in self.var_cfg["postprocessing"]["plots"]["cv_results_plot"]["metrics"]:
                    self.plotter.plot_cv_results_plots(
                        data_to_plot=self.cv_results_dct[metric],
                        rel=None
                    )
            else:
                raise ValueError("We must condense the cv results before creating the plot")

        if "create_shap_plots" in self.methods_to_apply:
            # TODO: we may do multiple calls with different parameters (paper plot, supplement, ia_values, etc)
            self.plotter.plot_shap_beeswarm_plots(prepare_data_func=self.shap_processor.prepare_data)

    def sanity_check_pred_vs_true(self):
        """
        This function analysis the predicted and the true criterion values within and across samples.
        We do this to further investigate unexpected predictive patterns in the mac analysis.
        To do so, we aggregate the predicted vs. true values across
            - repetitions
            - outer folds
            - imputations
        and compute some summary statistics and metrics
        """
        # Could in principle do this for all analyses
        root_dir = self.var_cfg["postprocessing"]["check_pred_vs_true"]["path"]

        # Walk through all subdirectories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if not dirnames:
                index_data = {}

                for filename in filenames:
                    reps_to_check = self.var_cfg["postprocessing"]["check_pred_vs_true"]["reps_to_check"]
                    if filename.startswith('pred_vs_true_rep_') and filename.endswith('.json'):
                        rep_number = filename.removeprefix('pred_vs_true_rep_').removesuffix('.json')

                        if rep_number.isdigit() and int(rep_number) in reps_to_check:
                            file_path = os.path.join(dirpath, filename)
                            with open(file_path, 'r') as f:
                                data = json.load(f)

                            # Iterate over the nested structure
                            for outer_fold_data in data.values():
                                for imp_data in outer_fold_data.values():
                                    for index, pred_true in imp_data.items():
                                        # Append the pred_true tuple to the index in index_data
                                        if index not in index_data:
                                            index_data[index] = []
                                        index_data[index].append(pred_true)

                # If index_data is not empty, process it
                if index_data:
                    sample_data = {}

                    # Process the collected data
                    for index, pred_true_list in index_data.items():
                        # Extract sample name from index (e.g., 'cocoesm' from 'cocoesm_7')
                        sample_name = index.split('_')[0]
                        if sample_name not in sample_data:
                            sample_data[sample_name] = {'pred': [], 'true': [], 'diff': []}
                        for pred_true in pred_true_list:
                            pred, true = pred_true
                            sample_data[sample_name]['pred'].append(pred)
                            sample_data[sample_name]['true'].append(true)
                            sample_data[sample_name]['diff'].append(true - pred)

                    dir_components = os.path.normpath(dirpath).split(os.sep)

                    self.plotter.plot_pred_true_parity(sample_data,
                                                       feature_combination="mac",
                                                       samples_to_include=dir_components[-3],
                                                       crit=dir_components[-2],
                                                       model=dir_components[-1]
                                                       )

                    # Compute summary statistics for each sample
                    summary_statistics = {}

                    for sample_name, values in sample_data.items():
                        pred_array = np.array(values['pred'])
                        true_array = np.array(values['true'])
                        diff_array = np.array(values['diff'])

                        # Compute R² and Spearman's rho if there are at least two data points
                        if len(pred_array) > 1:
                            r2 = r2_score(true_array, pred_array)
                            rho, _ = spearmanr(true_array, pred_array)
                        else:
                            r2 = None
                            rho = None

                        summary_statistics[sample_name] = {
                            'pred_mean': np.round(np.mean(pred_array), 4),
                            'pred_std': np.round(np.std(pred_array), 4),
                            'true_mean': np.round(np.mean(true_array), 4),
                            'true_std': np.round(np.std(true_array), 4),
                            'diff_mean': np.round(np.mean(diff_array), 4),
                            'diff_std': np.round(np.std(diff_array), 4),
                            'r2_score': np.round(r2, 4) if r2 is not None else None,
                            'spearman_rho': np.round(rho, 4) if rho is not None else None
                        }

                    # Save the summary statistics to a JSON file in the terminal directory
                    output_file = os.path.join(dirpath, 'pred_vs_true_summary.json')
                    with open(output_file, 'w') as f:
                        json.dump(summary_statistics, f, indent=4)

    def extract_metrics(self, base_dir, metric, cv_results_filename):
        """
        Extract required metrics from 'proc_cv_results.json' files in the directory structure.

        Args:
            base_dir (str): The base directory to start the search.
            metric (str): The metric to extract.

        Returns:
            dict: Extracted metrics dictionary.
            list: Data points for DataFrame creation.
        """
        metrics_dict = {}
        data_points = []

        for root, _, files in os.walk(base_dir):
            if cv_results_filename in files:

                rearranged_key, feature_combination, samples_to_include, crit, model\
                    = self.rearrange_path_parts(root, base_dir, min_depth=4)

                try:
                    with open(os.path.join(root, 'cv_results_summary.json'), 'r') as f:
                        proc_cv_results = json.load(f)

                    m_metric = proc_cv_results['m'][metric]
                    sd_metric = proc_cv_results['sd_across_folds_imps'][metric]

                    metrics_dict[rearranged_key] = {f'm_{metric}': m_metric, f'sd_{metric}': sd_metric}
                    data_points.append({
                        'crit': crit,
                        'model': model,
                        'samples_to_include': samples_to_include,
                        'feature_combination': feature_combination,
                        f"m_{metric}": m_metric
                    })

                except Exception as e:
                    print(f"Error reading {os.path.join(root, 'proc_cv_results.json')}: {e}")

        return metrics_dict, data_points

    def extract_coefficients(self, base_dir, coef_filename):
        """
        Extract top coefficients from 'lin_model_coefficients.json' files in the directory structure.

        Args:
            base_dir (str): The base directory to start the search.

        Returns:
            dict: Extracted coefficients dictionary.
            list: Coefficient points for DataFrame creation.
        """
        coefficients_dict = {}
        coefficient_points = []

        for root, _, files in os.walk(base_dir):
            if coef_filename in files:
                rearranged_key, feature_combination, samples_to_include, crit, model\
                    = self.rearrange_path_parts(root, base_dir, min_depth=4)

                try:
                    with open(os.path.join(root, 'lin_model_coefs_summary.json'), 'r') as f:
                        lin_model_coefficients = json.load(f)

                    coefficients = lin_model_coefficients['m']
                    sorted_coefficients = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_seven_coefficients = sorted_coefficients[:7]

                    coefficients_dict[rearranged_key] = dict(top_seven_coefficients)
                    coefficient_points.append({
                        'crit': crit,
                        'model': model,
                        'samples_to_include': samples_to_include,
                        'feature_combination': feature_combination,
                        'coefficients': top_seven_coefficients
                    })

                except Exception as e:
                    print(f"Error reading {os.path.join(root, 'proc_lin_model_coefficients.json')}: {e}")

        return coefficients_dict, coefficient_points

    @staticmethod
    def rearrange_path_parts(root, base_dir, min_depth=4):
        """
        Rearranges parts of a relative path if it meets the minimum depth requirement.

        Args:
            root (str): The full path to process.
            base_dir (str): The base directory to calculate the relative path from.
            min_depth (int): Minimum depth of the path to proceed.

        Returns:
            str or None: Rearranged path key if the depth requirement is met, else None.
        """
        relative_path = os.path.relpath(root, base_dir)
        path_parts = relative_path.strip(os.sep).split(os.sep)

        if len(path_parts) >= min_depth:
            rearranged_path_parts = '_'.join([path_parts[2], path_parts[0], path_parts[3], path_parts[1]])
            feature_combination = path_parts[0]
            samples_to_include = path_parts[1]
            crit = path_parts[2]
            model = path_parts[3]
            return (
                rearranged_path_parts,
                feature_combination,
                samples_to_include,
                crit,
                model
            )
        else:
            print(f"Skipping directory {root} due to insufficient path depth.")
            return None

    @staticmethod
    def create_df_table(data, metric, output_dir, custom_order):
        """
        Create DataFrame from metrics data, save to Excel.

        Args:
            data (list): Data points extracted for DataFrame.
            metric (str): The metric used for the heatmap title.
            output_dir (str): Directory to save the Excel file.
        """
        if data:
            filtered_data_points = [entry for entry in data if entry.get('samples_to_include') != 'control']
            df = pd.DataFrame(filtered_data_points)
            df.set_index(['crit', 'model', 'samples_to_include'], inplace=True)
            df_pivot = df.pivot_table(values=f"m_{metric}", index=['crit', 'model', 'samples_to_include'], columns='feature_combination', aggfunc=np.mean)

            # Round
            df_pivot = df_pivot.round(3)
            df_pivot = df_pivot.reindex(columns=custom_order)

            output_path = os.path.join(output_dir, f'cv_results_{metric}.xlsx')
            df_pivot.to_excel(output_path, merge_cells=True)

    @staticmethod
    def create_coefficients_dataframe(data, output_dir):
        """
        Create a DataFrame of the coefficients data and save it to Excel.

        Args:
            coefficient_points (list): Coefficient points extracted for DataFrame.
            output_dir (str): Directory to save the Excel file.

        Returns:
            None
        """
        if data:
            filtered_coeff_points = [entry for entry in data if entry.get('samples_to_include') != 'control']
            df_coeff = pd.DataFrame(filtered_coeff_points)
            df_coeff.set_index(['crit', 'model', 'samples_to_include'], inplace=True)
            df_coeff_pivot = df_coeff.pivot_table(values='coefficients', index=['crit', 'model', 'samples_to_include'], columns='feature_combination', aggfunc=lambda x: x)

            output_path = os.path.join(output_dir, 'df_coefficients.xlsx')
            df_coeff_pivot.to_excel(output_path, merge_cells=True)

    def defaultdict_to_dict(self, dct):
        """
        This function converts a nested default dict into a dict via recursion

        Args:
            dct: dict

        Returns:
            dict
        """
        if isinstance(dct, defaultdict):
            dct = {k: self.defaultdict_to_dict(v) for k, v in dct.items()}
        return dct




