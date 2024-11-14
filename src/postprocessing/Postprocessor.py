import copy
import json
import os
import re
from datetime import datetime
import numpy as np

from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.utils.DataLoader import DataLoader

import pandas as pd

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

        self.logger = Logger(log_dir=self.var_cfg["general"]["log_dir"], log_file=self.var_cfg["general"]["log_name"])
        self.data_loader = DataLoader()
        self.descriptives_creator = DescriptiveStatistics(fix_cfg=self.fix_cfg, var_cfg=self.var_cfg, name_mapping=name_mapping)
        self.significance_testing = SignificanceTesting(var_cfg=self.var_cfg)

        self.base_result_dir = self.var_cfg["postprocessing"]["raw_results_path"]
        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]
        self.processed_output_path = self.var_cfg["postprocessing"]["processed_results_path"]
        self.plotter = ResultPlotter(var_cfg=self.var_cfg, plot_base_dir=self.processed_output_path)
        self.cv_result_dct = {}

        self.shap_processor = ShapProcessor(
            var_cfg=self.var_cfg,
            base_result_dir=self.base_result_dir,
            processed_output_path=self.processed_output_path,
            name_mapping=self.name_mapping,
        )
        self.feature_mappings = {}
        self.metric = self.var_cfg["postprocessing"]["metric"]
        self.methods_to_apply = self.var_cfg["postprocessing"]["methods"]

    def postprocess(self):
        """
        This is kind of a wrapper method that does all the postprocessing steps specified in the config.
        It may invoke the methods in the other classes

        Returns:

        """
        self.create_correction_mapping()

        # These are the costly operations that condenses the inforfmation of the full cluster results
        if "process_cv_results" in self.methods_to_apply:
            self.process_cv_results()

        if "process_lin_model_coefs" in self.methods_to_apply:
            self.process_lin_model_coefs()

        if "process_shap_values" in self.methods_to_apply:
            self.shap_processor.aggregate_shap_values(feature_mappings=self.feature_mappings)
            if self.var_cfg["postprocessing"]["shap_processing"]["merge_folders"]:  # This applies to already processed folders
                self.shap_processor.merge_folders(source1=self.var_cfg["postprocessing"]["shap_processing"]["source_1"],
                                                  source2=self.var_cfg["postprocessing"]["shap_processing"]["source_2"],
                                                  merged_folder=self.var_cfg["postprocessing"]["shap_processing"]["output_folder"]
                                                  )

        # These are the operations to condense the results into one file (.e.g., of the coefficients
        if "condense_results" in self.methods_to_apply:
            self.extract_fitting_times(base_dir=self.base_result_dir, output_dir=self.processed_output_path)
            metrics_dict, data_points = self.extract_metrics(self.processed_output_path, self.metric)
            coefficients_dict, coefficient_points = self.extract_coefficients(self.processed_output_path)
            self.create_df_table(data_points, self.metric, self.processed_output_path)
            self.create_coefficients_dataframe(coefficient_points, self.processed_output_path)

        if "create_shap_plots" in self.methods_to_apply:
            # self.shap_processor.prepare_shap_plot_data()
            self.plotter.plot_shap_beeswarm_plots(prepare_data_func=self.shap_processor.prepare_data)

        if "create_descriptives" in self.methods_to_apply:
            self.descriptives_creator.create_m_sd_feature_table()
            self.descriptives_creator.create_wb_item_statistics()

        if "conduct_significance_tests" in self.methods_to_apply:
            self.significance_testing.significance_testing(dct=self.cv_result_dct.copy())

    def create_correction_mapping(self):
        """
        We need to map back shap_values and lin_model_coefs for analysis where country-level and individual-level
        features are involved. To do so, we load the data, extract the features for each feature_combination affected
        and create a mapping

        Note: We need this only for the linear model coefficients, not the SHAP values
        """
        df = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        for feat_combo in ["pl_mac", "pl_mac_nnse", "pl_srmc_mac", "pl_srmc_mac_nnse", "all_in", "all_in_nnse"]:
            selected_columns = []
            feature_prefix_lst = feat_combo.split("_")
            if feat_combo in ["all_in", "all_in_nnse"]:
                feature_prefix_lst = ["pl", "srmc", "sens", "mac"]
            # select cols
            for feature_cat in feature_prefix_lst:
                for col in df.columns:
                    if col.startswith(feature_cat):
                        selected_columns.append(col)
            # create mappings
            col_order_before_imputation = selected_columns
            mac_columns = [col for col in selected_columns if col.startswith("mac_")]
            other_columns = [col for col in selected_columns if not col.startswith("mac_")]
            col_order_after_imputation = mac_columns + other_columns
            mapping = {col_before: col_after for col_after, col_before in
                       zip(col_order_after_imputation, col_order_before_imputation)}
            assert len(mapping) == len(col_order_before_imputation)
            getattr(self, "feature_mappings")[feat_combo] = mapping

    @staticmethod
    def extract_fitting_times(base_dir, output_dir):
        """
        Traverse the folder structure from a base directory, find log files with 'rank0',
        extract fitting times, and store results in a JSON file.

        Args:
            base_dir (str): The base directory to start the search.
            output_json_path (str): The path to save the output JSON file.
        """
        fitting_times_dict = {}
        # Regular expressions to match the fitting time and timestamp pattern
        time_pattern = re.compile(r"repeated_nested_cv executed in (\d+h \d+m \d+\.\d+s) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

        # Traverse the directory structure
        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                # Look for log files that contain 'rank0'
                if "rank0" in file_name:
                    file_path = os.path.join(root, file_name)
                    latest_time_str = None
                    latest_timestamp = None

                    try:
                        with open(file_path, 'r') as file:
                            content = file.read()

                        # Search for all fitting times and timestamps in the file
                        matches = time_pattern.findall(content)
                        for match in matches:
                            time_str, timestamp_str = match
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                            # Check if this is the latest timestamp
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                                latest_time_str = time_str

                        # If a fitting time is found, store it
                        if latest_time_str:
                            # Convert directory structure to a concatenated key
                            relative_path = os.path.relpath(root, base_dir)
                            concatenated_key = relative_path.replace(os.sep, "_")

                            # Store the latest extracted time in the dict
                            fitting_times_dict[concatenated_key] = latest_time_str

                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

        # Sort the dictionary by keys (alphabetically)
        fitting_times_dict = dict(sorted(fitting_times_dict.items()))

        # Save the result to a JSON file
        output_json_path = os.path.join(output_dir, "fit_times.json")
        with open(output_json_path, 'w') as json_file:
            json.dump(fitting_times_dict, json_file, indent=4)

        print(f"Fitting times extracted and saved to {output_json_path}")

    @staticmethod
    def extract_metrics(base_dir, metric):
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
            if 'proc_cv_results.json' in files:
                relative_path = os.path.relpath(root, base_dir)
                path_parts = relative_path.strip(os.sep).split(os.sep)

                if len(path_parts) >= 4:
                    a, b, c, d = path_parts[0], path_parts[1], path_parts[2], path_parts[3]
                else:
                    print(f"Skipping directory {root} due to insufficient path depth.")
                    continue

                rearranged_key = '_'.join([c, a, d, b])

                try:
                    with open(os.path.join(root, 'proc_cv_results.json'), 'r') as f:
                        proc_cv_results = json.load(f)

                    m_metric = proc_cv_results['m'][metric]
                    sd_metric = proc_cv_results['sd_across_folds_imps'][metric]

                    metrics_dict[rearranged_key] = {f'm_{metric}': m_metric, f'sd_{metric}': sd_metric}
                    data_points.append({'c': c, 'd': d, 'b': b, 'a': a, f"m_{metric}": m_metric})

                except Exception as e:
                    print(f"Error reading {os.path.join(root, 'proc_cv_results.json')}: {e}")

        return metrics_dict, data_points

    @staticmethod
    def extract_coefficients(base_dir):
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
            if 'proc_lin_model_coefficients.json' in files:
                relative_path = os.path.relpath(root, base_dir)
                path_parts = relative_path.strip(os.sep).split(os.sep)

                if len(path_parts) >= 4:
                    a, b, c, d = path_parts[0], path_parts[1], path_parts[2], path_parts[3]
                else:
                    print(f"Skipping directory {root} due to insufficient path depth.")
                    continue

                rearranged_key = '_'.join([c, a, d, b])

                try:
                    with open(os.path.join(root, 'proc_lin_model_coefficients.json'), 'r') as f:
                        lin_model_coefficients = json.load(f)

                    coefficients = lin_model_coefficients['m']
                    sorted_coefficients = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_seven_coefficients = sorted_coefficients[:7]

                    coefficients_dict[rearranged_key] = dict(top_seven_coefficients)
                    coefficient_points.append({
                        'c': c, 'd': d, 'b': b, 'a': a,
                        'coefficients': top_seven_coefficients
                    })

                except Exception as e:
                    print(f"Error reading {os.path.join(root, 'proc_lin_model_coefficients.json')}: {e}")

        return coefficients_dict, coefficient_points

    @staticmethod
    def create_df_table(data_points, metric, output_dir):
        """
        Create DataFrame from metrics data, save to Excel.

        Args:
            data_points (list): Data points extracted for DataFrame.
            metric (str): The metric used for the heatmap title.
            output_dir (str): Directory to save the Excel file.

        Returns:
            None
        """
        if data_points:
            filtered_data_points = [entry for entry in data_points if entry.get('b') != 'control']
            df = pd.DataFrame(filtered_data_points)
            df.set_index(['c', 'd', 'b'], inplace=True)
            df_pivot = df.pivot_table(values=f"m_{metric}", index=['c', 'd', 'b'], columns='a', aggfunc=np.mean)

            custom_order = [
                "pl", "pl_nnse", "srmc", "sens", "mac",
                "pl_srmc", "pl_srmc_nnse", "pl_sens", "pl_sens_nnse",
                "pl_srmc_sens", "pl_srmc_sens_nnse", "srmc_control",
                "pl_srmc_control", "pl_mac", "pl_mac_nnse",
                "pl_srmc_mac", "pl_srmc_mac_nnse", "all_in", "all_in_nnse"
            ]
            df_pivot = df_pivot.reindex(columns=custom_order)

            output_path = os.path.join(output_dir, f'df_pivot_{metric}.xlsx')
            df_pivot.to_excel(output_path, merge_cells=True)

    @staticmethod
    def create_coefficients_dataframe(coefficient_points, output_dir):
        """
        Create a DataFrame of the coefficients data and save it to Excel.

        Args:
            coefficient_points (list): Coefficient points extracted for DataFrame.
            output_dir (str): Directory to save the Excel file.

        Returns:
            None
        """
        if coefficient_points:
            filtered_coeff_points = [entry for entry in coefficient_points if entry.get('b') != 'control']
            df_coeff = pd.DataFrame(filtered_coeff_points)
            df_coeff.set_index(['c', 'd', 'b'], inplace=True)
            df_coeff_pivot = df_coeff.pivot_table(values='coefficients', index=['c', 'd', 'b'], columns='a', aggfunc=lambda x: x)

            output_path = os.path.join(output_dir, 'df_coefficients.xlsx')
            df_coeff_pivot.to_excel(output_path, merge_cells=True)

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
        """
        cv_result_dct = self.get_results(result_type="performance")
        self.cv_result_dct = cv_result_dct
        cv_result_dct_processed = self.compute_result_statistics(result_dict=cv_result_dct, result_type="performance")
        self.store_results(processed_result_dct=cv_result_dct_processed, result_type="performance")

    def process_lin_model_coefs(self):
        """
        This function
            - iterates over all subdirectories of self.base_result_dir
            - stores the content of the result files "lin_model_coefficients.json" in a dictionary that mirros the folder structure
            - computes M
            - computes SD
                1) across outer folds and imputations
                2) across outer folds within imputations
                3) across imputations within outer folds
        """
        coef_result_dct = self.get_results(result_type="lin_model_coefs")
        coef_dct_processed = self.compute_result_statistics(result_dict=coef_result_dct, result_type="lin_model_coefs")
        self.store_results(processed_result_dct=coef_dct_processed, result_type="lin_model_coefs")

    def get_results(self, result_type: str) -> dict:
        """
        This function:
            - Iterates over all subdirectories of self.base_result_dir
            - Stores the content of the result files "cv_results.json" in a dictionary that mirrors the folder structure

        Args:
            result_type (str): specifying the result type, should be "performance" or "lin_model_coefs"
        Returns:
            dict: Nested dict containing the cv_results of all analysis in self.base_result_dir
        """
        results_dict = {}  # Initialize the dictionary to store results
        try:
            results_file_name = self.result_filenames[result_type]
        except KeyError:
            raise ValueError(f"Result type {result_type} not recognized, must be 'performance' or 'lin_model_coefs'")

        for root, dirs, files in os.walk(self.base_result_dir):
            if results_file_name in files:
                # Build the relative path from base_result_dir to cv_results.json's directory
                rel_path = os.path.relpath(root, self.base_result_dir)
                # Split the relative path into parts
                path_parts = rel_path.split(os.sep)
                # The path_parts list will be used to build the nested dictionary

                # Load the result_file
                cv_results_path = os.path.join(root, str(results_file_name))
                cv_results = self.data_loader.read_json(cv_results_path)

                # Build the nested dictionary structure
                current_level = results_dict
                for part in path_parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                # At this point, current_level is the innermost dictionary corresponding to the path
                current_level[str(results_file_name[:-5])] = cv_results

        return results_dict

    def compute_result_statistics(self, result_dict: dict, result_type: str) -> dict:
        """
        This function computes statistics of the results obtained, it
            - computes the mean (m) across all 500 outer folds
            - computes the standard deviation (sd)
                1) across outer folds and imputations
                2) across outer folds within imputations
                3) across imputations within outer folds

        Args:
            result_dict (dict): Nested dict containing the result of all analysis in self.base_result_dir
            result_type (str): Specifying the result type, should be "performance" or "lin_model_coefs"

        Returns:
            dict: Nested dict containing the m and sd calculated
        """
        # Create a deep copy to avoid modifying the original dictionary
        result_dict_copy = copy.deepcopy(result_dict)

        if result_type == "performance":
            key_to_search = "cv_results"
        elif result_type == "lin_model_coefs":
            key_to_search = "lin_model_coefficients"
        else:
            raise ValueError(f"Result type {result_type} not recognized, must be 'performance' or 'lin_model_coefs'")

        # Recursive function to traverse and process the dictionary
        def process_node(node):
            for key in list(node.keys()):
                if key == key_to_search:
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
                    df = self.correct_mac_feature_assign(df)

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

                    # Create a new key in the current node for the results, replacing 'cv_results'
                    node[key_to_search] = {
                        'm': group_mean,
                        'sd_across_folds_imps': sd_across_folds_imps,
                        'sd_within_folds_across_imps': sd_within_folds_across_imps,
                        'sd_across_folds_within_imps': sd_across_folds_within_imps
                    }

                    # Add non-zero coefs of features if result_type is "lin_model_coefs"
                    if result_type == "lin_model_coefs":
                        print()
                        non_zero_count = (df[metric_cols] != 0).sum().to_dict()
                        node[key_to_search]['coef_not_zero_across_folds_imps'] = non_zero_count

                        # Sort based on the absolute value of group_mean
                        sorted_keys = sorted(group_mean.keys(), key=lambda k: abs(group_mean[k]), reverse=True)

                        # Reorder the dicts based on the sorted keys
                        node[key_to_search]['coef_not_zero_across_folds_imps'] = {k: non_zero_count[k] for k in sorted_keys}
                        node[key_to_search]['m'] = {k: group_mean[k] for k in sorted_keys}
                        node[key_to_search]['sd_across_folds_imps'] = {k: sd_across_folds_imps[k] for k in sorted_keys}
                        node[key_to_search]['sd_within_folds_across_imps'] = {k: sd_within_folds_across_imps[k] for k in sorted_keys}
                        node[key_to_search]['sd_across_folds_within_imps'] = {k: sd_across_folds_within_imps[k] for k in sorted_keys}

                elif isinstance(node[key], dict):
                    # Recursively process the sub-dictionary
                    process_node(node[key])

        # Start processing from the root of the copied dictionary
        process_node(result_dict_copy)

        return result_dict_copy

    def correct_mac_feature_assign(self, df):
        """
        This function applies a correction to the feature assignment in the case where country-level and individual level
        features are together in a certain analysis

        Args:
            df:

        Returns:
            df:

        """
        cols = df.columns[3:]
        for feat_combo, mapping in self.feature_mappings.items():
            if len(cols) == len(mapping):
                df = df.rename(columns=mapping)
            else:
                continue
        return df

    def store_results(self, processed_result_dct: dict, result_type: str) -> None:
        """
        This function stores the results contained by the given dict.
        It mirrors the dict structure in the folder structure up to the given key (results_key).
        The file should be stored under the name 'results_proc' as JSON files.

        Parameters:
        - processed_result_dct: The dictionary containing the processed results to store.
        - result_type: The type of result ('performance' or 'lin_model_coefs').

        Returns:
        - None
        """
        try:
            # Extract the raw filename and derive the key and processed filename
            results_raw_filename = self.var_cfg["analysis"]["output_filenames"][result_type]
            results_key = results_raw_filename.split('.')[0]
            results_proc = f"proc_{results_raw_filename}"  # JSON file extension

            # Define the base directory where the results will be stored
            base_output_dir = self.processed_output_path

            # Recursive function to traverse and find the 'results_key'
            def store_nested_dict(dct, current_path):
                for key, value in dct.items():
                    # If we reach the 'results_key', save the associated value as a JSON file
                    if key == results_key:
                        output_file = os.path.join(current_path, results_proc)
                        # Store the dictionary under 'results_key' as a JSON file
                        with open(output_file, 'w') as f:
                            json.dump(value, f, indent=4)
                        print(f"Results successfully stored in: {output_file}")
                        return  # Stop further recursion when we reach the target key

                    # If value is a dictionary, recurse further and create a folder
                    if isinstance(value, dict):
                        # Create the folder for the current key
                        new_path = os.path.join(current_path, key)
                        os.makedirs(new_path, exist_ok=True)
                        # Recursively go deeper into the dictionary
                        store_nested_dict(value, new_path)

            # Start the recursion from the base directory
            store_nested_dict(processed_result_dct, base_output_dir)

        except KeyError:
            raise ValueError(f"Result type '{result_type}' not recognized, must be 'performance' or 'lin_model_coefs'")
        except Exception as e:
            raise RuntimeError(f"An error occurred while storing the results: {e}")




