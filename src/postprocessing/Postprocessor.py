import copy
import json
import os
import re
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

from src.postprocessing.ResultPlotter import ResultPlotter
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
        self.plotter = ResultPlotter(var_cfg=self.var_cfg)
        self.base_result_dir = self.var_cfg["analysis"]["results_cluster_path"]
        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]

        self.processed_output_path = self.var_cfg["postprocessing"]["processed_output_path"]

        self.cv_result_dct = {}

        self.shap_processor = ShapProcessor(
            var_cfg=self.var_cfg,
            base_result_dir=self.base_result_dir,
            processed_output_path=self.processed_output_path,
            name_mapping=self.name_mapping,
        )

    def postprocess(self):
        #self.extract_fitting_times(base_dir=self.base_result_dir, output_dir=self.processed_output_path)
        #self.shap_processor.aggregate_shap_values()
        #self.shap_processor.prepare_importance_plot_data()
        #self.plotter.plot_shap_importance_plot(
        #    data=self.shap_processor.data_importance_plot,
        #    crit="state_wb",
        #    samples_to_include="selected",
        #    model="elasticnet"
        #)
        #self.plotter.plot_shap_beeswarm_plot(
        #    data=self.shap_processor.data_importance_plot,
        #    crit="state_wb",
        #    samples_to_include="selected",
        #    model="elasticnet"
        #)
        #self.process_cv_results()
        self.process_lin_model_coefs()
        #self.extract_metrics_and_coefficients(base_dir=self.processed_output_path, output_dir=self.processed_output_path)
        # self.shap_processor.recreate_explanation_objects()
        #self.descriptives_creator.create_m_sd_feature_table()
        #self.descriptives_creator.create_wb_item_statistics()
        #self.significance_testing.compare_models(dct=self.cv_result_dct.copy())
        #self.significance_testing.significance_testing(dct=self.cv_result_dct.copy())

    @staticmethod
    def extract_fitting_times(base_dir, output_dir):
        """
        Traverse the folder structure from a base directory, find log files with 'rank0',
        extract fitting times, and store results in a JSON file.

        Args:
            base_dir (str): The base directory to start the search.
            output_json_path (str): The path to save the output JSON file.

        Returns:
            None
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
    def extract_metrics_and_coefficients(base_dir, output_dir):
        """
        Traverse the folder structure from a base directory, find 'proc_cv_results.json' and 'lin_model_coefficients.json',
        extract required metrics and coefficients, store results in JSON files, and generate a heatmap.

        Args:
            base_dir (str): The base directory to start the search.
            output_dir (str): The directory to save the output JSON files.

        Returns:
            None
        """
        metrics_dict = {}
        coefficients_dict = {}
        data_points = []
        coefficient_points = []

        # Traverse the directory structure
        for root, dirs, files in os.walk(base_dir):
            # Check if 'proc_cv_results.json' exists in the current directory
            if 'proc_cv_results.json' in files:
                # Build the key from the folder structure rearranged from a/b/c/d/ to c/a/d/b
                relative_path = os.path.relpath(root, base_dir)
                path_parts = relative_path.strip(os.sep).split(os.sep)

                if len(path_parts) >= 4:
                    a = path_parts[0]
                    b = path_parts[1]
                    c = path_parts[2]
                    d = path_parts[3]
                else:
                    # Handle cases where path_parts has fewer than 4 elements
                    print(f"Skipping directory {root} due to insufficient path depth.")
                    continue

                # Rearrange the parts according to c/a/d/b for the concatenated key
                rearranged_parts = [c, a, d, b]
                concatenated_key = '_'.join(rearranged_parts)

                # Extract metrics from 'proc_cv_results.json'
                proc_cv_results_path = os.path.join(root, 'proc_cv_results.json')
                try:
                    with open(proc_cv_results_path, 'r') as f:
                        proc_cv_results = json.load(f)

                    # Extract 'm' and 'sd' of 'r2' metric
                    m_r2 = proc_cv_results['m']['r2']
                    sd_r2 = proc_cv_results['sd_across_folds_imps']['r2']  # Adjust the key if needed

                    # Store in metrics_dict
                    metrics_dict[concatenated_key] = {'m_r2': m_r2, 'sd_r2': sd_r2}

                    # Collect data for DataFrame
                    data_points.append({'c': c, 'd': d, 'b': b, 'a': a, 'm_r2': m_r2})

                except Exception as e:
                    print(f"Error reading {proc_cv_results_path}: {e}")

                # Check if 'lin_model_coefficients.json' exists in the current directory
                if 'proc_lin_model_coefficients.json' in files:
                    lin_model_coefficients_path = os.path.join(root, 'proc_lin_model_coefficients.json')
                    try:
                        with open(lin_model_coefficients_path, 'r') as f:
                            lin_model_coefficients = json.load(f)

                        # Extract 'm' coefficients
                        coefficients = lin_model_coefficients['m']

                        # Get the five most important coefficients (by absolute value)
                        sorted_coefficients = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
                        top_seven_coefficients = sorted_coefficients[:7]  # Get the top 7

                        # Store the top 7 coefficients in coefficients_dict
                        coefficients_dict[concatenated_key] = dict(top_seven_coefficients)

                        # Collect data for DataFrame with tuples of (feature, value)
                        coefficient_points.append({
                            'c': c,
                            'd': d,
                            'b': b,
                            'a': a,
                            'coefficients': top_seven_coefficients  # Store as list of tuples
                        })

                    except Exception as e:
                        print(f"Error reading {lin_model_coefficients_path}: {e}")

        # Sort the dictionaries by keys (alphabetically)
        metrics_dict = dict(sorted(metrics_dict.items()))
        coefficients_dict = dict(sorted(coefficients_dict.items()))

        # Save the results to JSON files
        os.makedirs(output_dir, exist_ok=True)

        metrics_output_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_output_path, 'w') as json_file:
            json.dump(metrics_dict, json_file, indent=4)

        coefficients_output_path = os.path.join(output_dir, 'coefficients.json')
        with open(coefficients_output_path, 'w') as json_file:
            json.dump(coefficients_dict, json_file, indent=4)

        print(f"Metrics extracted and saved to {metrics_output_path}")
        print(f"Coefficients extracted and saved to {coefficients_output_path}")

        # Create DataFrame from data_points
        if data_points:

            # Filter out control analyses
            filtered_data_points = [entry for entry in data_points if entry.get('b') != 'control']

            df = pd.DataFrame(filtered_data_points)

            # Set MultiIndex from 'c', 'd', 'b' (in that order)
            df.set_index(['c', 'd', 'b'], inplace=True)

            # Pivot the DataFrame to have 'a' as columns, and the MultiIndex as rows
            df_pivot = df.pivot_table(values='m_r2', index=['c', 'd', 'b'], columns='a', aggfunc=np.mean)

            df_pivot = df_pivot.sort_index()
            # Fill missing values with np.nan (this is default behavior)
            custom_order = ["pl", "srmc", "sens", "mac", "pl_srmc", "pl_sens", "pl_srmc_sens", "pl_mac", "pl_srmc_mac", "all_in"]  # Replace with the actual column names in the desired order
            # Reindex df_pivot to match the custom column ord
            df_pivot = df_pivot.reindex(columns=custom_order)

            df_pivot.to_excel("df_pivot.xlsx", merge_cells=True)


            # Display the DataFrame
            print("\nDataFrame of m_r2 values:")
            print(df_pivot)

            # Generate and display the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_pivot, annot=True, cmap='viridis', fmt=".3f")
            plt.title('Heatmap of m_r2 Values')
            plt.ylabel('c / d / b')
            plt.xlabel('a')
            plt.tight_layout()
            plt.show()
        else:
            print("No data available to create DataFrame and heatmap.")

        # Process the coefficients DataFrame
        if coefficient_points:
            filtered_coeff_points = [entry for entry in coefficient_points if entry.get('b') != 'control']
            df_coeff = pd.DataFrame(filtered_coeff_points)
            df_coeff.set_index(['c', 'd', 'b'], inplace=True)

            # Pivot table with 'a' as columns and each entry containing the top 7 coefficients as tuples
            df_coeff_pivot = df_coeff.pivot_table(values='coefficients', index=['c', 'd', 'b'], columns='a', aggfunc=lambda x: x)
            df_coeff_pivot = df_coeff_pivot.sort_index()
            df_coeff_pivot.to_excel("df_coefficients.xlsx", merge_cells=True)

            print("\nDataFrame of Top 7 Coefficients:")
            print(df_coeff_pivot)

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
        Returns:
        """
        cv_result_dct = self.get_results(result_type="performance")
        self.cv_result_dct = cv_result_dct
        cv_result_dct_processed = self.compute_result_statistics(result_dict=cv_result_dct, result_type="performance")
        self.store_results(processed_result_dct=cv_result_dct_processed, result_type="performance")

        # self.plotter.plot_cv_results(cv_result_dct_processed)

    def process_lin_model_coefs(self):
        """

        Returns:

        """
        coef_result_dct = self.get_results(result_type="lin_model_coefs")
        coef_dct_processed = self.compute_result_statistics(result_dict=coef_result_dct, result_type="lin_model_coefs")
        coef_dct_processed = self.reorder_mac_keys_in_nested_dict(input_dict=coef_dct_processed)
        self.store_results(processed_result_dct=coef_dct_processed, result_type="lin_model_coefs")

    def reorder_mac_keys_in_nested_dict(self, input_dict):
        """
        Searches for the 'lin_model_coefficients' key in a nested dictionary and reorders its keys so that
        keys starting with 'mac_' are moved to the beginning.

        Args:
            input_dict (dict): The nested dictionary to process.

        Returns:
            dict: The modified dictionary with 'mac_' keys reordered within 'lin_model_coefficients'.
        """
        # Traverse the dictionary to find 'lin_model_coefficients'
        for key, value in input_dict.items():
            if key == "lin_model_coefficients" and isinstance(value, dict):
                # Found 'lin_model_coefficients', reorder its keys
                input_dict[key] = self.reorder_mac_keys(value)
            elif isinstance(value, dict):
                # Recurse into nested dictionaries
                self.reorder_mac_keys_in_nested_dict(value)

        return input_dict

    def reorder_mac_keys(self, input_dict):
        """
        Reorders keys in the input dictionary so that keys starting with "mac_" are moved to the beginning.

        Args:
            input_dict (dict): The dictionary to reorder.

        Returns:
            dict: A reordered dictionary with "mac_" keys first.
        """
        # Separate keys that start with "mac_" and those that don't
        mac_keys = {k: v for k, v in input_dict.items() if k.startswith("mac_")}
        other_keys = {k: v for k, v in input_dict.items() if not k.startswith("mac_")}

        # Combine dictionaries with "mac_" keys first
        reordered_dict = {**other_keys, **mac_keys}

        return reordered_dict

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




