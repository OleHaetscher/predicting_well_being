import copy
import json
import os
import pickle
import re
import shutil
from collections import defaultdict
from datetime import datetime
import numpy as np

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
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

        if "merge_cluster_results" in self.methods_to_apply:
            self.merge_folders(source1=self.var_cfg["postprocessing"]["merge_cluster_results"]["source_1"],
                               source2=self.var_cfg["postprocessing"]["merge_cluster_results"]["source_2"],
                               merged_folder=self.var_cfg["postprocessing"]["merge_cluster_results"]["output_folder"]
                               )

        if "merge_reps" in self.methods_to_apply:  # If we run single analysis per rep on the cluster
            self.merge_cv_results_across_reps()
            self.merge_lin_model_coefs_across_reps()
            self.merge_shap_values_across_reps()
            #self.process_shap_ia_values()

        if "check_crit_dist" in self.methods_to_apply:
            self.check_crit_distribution_sample()
            self.check_crit_distribution_country()
            # self.check_crit_distribution_by_year()

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
            #self.extract_fitting_times(base_dir=self.base_result_dir, output_dir=self.processed_output_path)
            metrics_dict, data_points = self.extract_metrics(self.processed_output_path, self.metric)
            coefficients_dict, coefficient_points = self.extract_coefficients(self.processed_output_path)
            self.create_df_table(data_points, self.metric, self.processed_output_path)
            self.create_coefficients_dataframe(coefficient_points, self.processed_output_path)

        if "create_descriptives" in self.methods_to_apply:
            # pass
            rel = self.descriptives_creator.compute_rel()
            #self.descriptives_creator.create_m_sd_feature_table()
            #self.descriptives_creator.create_wb_item_statistics()

        if "create_cv_results_plots" in self.methods_to_apply:
            self.plotter.plot_figure_2(data_to_plot=metrics_dict, rel=rel)

        if "create_shap_plots" in self.methods_to_apply:
            # self.shap_processor.prepare_shap_plot_data()
            self.plotter.plot_shap_beeswarm_plots(prepare_data_func=self.shap_processor.prepare_data)

        if "conduct_significance_tests" in self.methods_to_apply:
            self.significance_testing.significance_testing(dct=self.cv_result_dct.copy())

    def merge_folders(self, source1: str, source2: str, merged_folder: str):
        """
        This is used to merge folders with processed shap_values to account for data from different sources. We do this
        here as copying the unprocessed shap values may consume to much storage

        Args:
            source1:
            source2:
            merged_folder:

        """
        # Create the merged folder if it doesn't exist
        os.makedirs(merged_folder, exist_ok=True)

        # Step 1: Copy all contents from source1 into merged_folder
        for root, dirs, files in os.walk(source1):
            # Get relative path for the current directory
            rel_path = os.path.relpath(root, source1)
            target_dir = os.path.join(merged_folder, rel_path)

            # Create directories as needed in the target location
            os.makedirs(target_dir, exist_ok=True)

            # Copy files to the merged folder
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(target_dir, file)
                shutil.copy2(src_file, dest_file)

        # Step 2: Process files from source2
        for root, dirs, files in os.walk(source2):
            rel_path = os.path.relpath(root, source2)
            target_dir = os.path.join(merged_folder, rel_path)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(target_dir, file)

                # Check if it's a log file
                if "log" in file:
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                else:
                    # Check for .json and .pkl files in case of conflicts
                    if os.path.isfile(dest_file) and (file.endswith(".json") or file.endswith(".pkl")):
                        # Conflict detected; keeping version from source1
                        print(f"Conflict detected for {os.path.relpath(dest_file, merged_folder)}. Keeping version from {source1}.")
                    else:
                        # Copy file from source2 if no conflict or if it's a new file
                        os.makedirs(target_dir, exist_ok=True)
                        shutil.copy2(src_file, dest_file)

        print(f"Merging complete. Results stored in {merged_folder}")

    def check_crit_distribution_sample(self):
        """
        This function calculates the mean (M) and standard deviation (SD)
        of the criteria across all samples and visualizes them in single
        plots per criterion.
        """

        # Load the data
        df = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        # Defragment the DataFrame to avoid performance issues
        df = df.copy()

        # Define samples and criteria
        samples = ["cocoesm", "cocout", "cocoms", "emotions", "pia", "zpid"]
        criteria = ["crit_state_pa", "crit_state_na", "crit_state_wb"]

        # Initialize a list to collect results
        results = []

        for sample in samples:
            # Filter rows where the index starts with the sample name
            sample_df = df[df.index.astype(str).str.startswith(sample)]

            for criterion in criteria:
                # Check if the criterion exists in the sample_df
                if criterion in sample_df.columns:
                    # Calculate mean and SD for the sample
                    mean_value = sample_df[criterion].mean()
                    std_value = sample_df[criterion].std()

                    # Append results as a dictionary
                    results.append({
                        'sample': sample,
                        'criterion': criterion,
                        'mean': mean_value,
                        'std': std_value
                    })
                else:
                    print(f"Criterion '{criterion}' not found in sample '{sample}'. Skipping.")

        # Create a DataFrame from the results
        result_df = pd.DataFrame(results)

        # Visualization: One plot for each criterion
        for criterion in criteria:
            # Filter data for the current criterion
            criterion_data = result_df[result_df['criterion'] == criterion]

            # Plot
            plt.figure(figsize=(8, 5))
            plt.bar(criterion_data['sample'], criterion_data['mean'], yerr=criterion_data['std'], capsize=5)
            plt.title(f"Mean and SD of {criterion}")
            plt.ylabel("Mean Value")
            plt.xlabel("Sample")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        print()

    def check_crit_distribution_country(self):
        """
        This function calculates the mean (M) and standard deviation (SD)
        of the criteria across countries (Germany, US, Other) and visualizes
        them in single plots per criterion.
        """

        # Load the data
        df = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        # Defragment the DataFrame to avoid performance issues
        df = df.copy()

        # Map countries to 'Germany', 'US', and 'Other'
        df['country_grouped'] = df['other_country'].apply(
            lambda x: x.lower() if x.lower() in ['germany', 'usa'] else 'other'
        )

        # Define criteria
        criteria = ["crit_state_pa", "crit_state_na", "crit_state_wb"]

        # Initialize a list to collect results
        results = []

        for criterion in criteria:
            # Check if the criterion exists in the DataFrame
            if criterion in df.columns:
                # Group by country and calculate mean and SD
                stats = df.groupby('country_grouped')[criterion].agg(['mean', 'std']).reset_index()
                stats['criterion'] = criterion  # Add the criterion name to the results

                # Append to results
                results.append(stats)
            else:
                print(f"Criterion '{criterion}' not found in the DataFrame. Skipping.")

        # Create a DataFrame from the results
        if results:
            result_df = pd.concat(results, ignore_index=True)
        else:
            print("No data available for the specified criteria.")
            return None

        # Visualization: One plot for each criterion
        for criterion in criteria:
            # Filter data for the current criterion
            criterion_data = result_df[result_df['criterion'] == criterion]

            # Plot
            plt.figure(figsize=(8, 5))
            plt.bar(criterion_data['country_grouped'], criterion_data['mean'],
                    yerr=criterion_data['std'], capsize=5)
            plt.title(f"Mean and SD of {criterion} by Country")
            plt.ylabel("Mean Value")
            plt.xlabel("Country")
            plt.tight_layout()
            plt.show()

        # Return the DataFrame with mean and SD
        return result_df

    def check_crit_distribution_by_year(self):
        """
        This function calculates the mean (M) and standard deviation (SD)
        of the criteria across all years and visualizes them in single
        plots per criterion. Samples with multiple years in 'other_years_of_participation'
        are excluded.
        """

        # Load the data
        df = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        # Defragment the DataFrame to avoid performance issues
        df = df.copy()

        # Define criteria
        criteria = ["crit_state_pa", "crit_state_na", "crit_state_wb"]

        # Filter out rows where 'other_years_of_participation' contains multiple years
        df = df[
            df['other_years_of_participation'].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit()))]

        # Convert 'other_years_of_participation' to numeric for grouping
        df['other_years_of_participation'] = pd.to_numeric(df['other_years_of_participation'])

        # Initialize a list to collect results
        results = []

        # Group by year
        grouped = df.groupby('other_years_of_participation')

        for year, group in grouped:
            for criterion in criteria:
                # Check if the criterion exists in the group
                if criterion in group.columns:
                    # Calculate mean and SD for the year
                    mean_value = group[criterion].mean()
                    std_value = group[criterion].std()

                    # Append results as a dictionary
                    results.append({
                        'year': year,
                        'criterion': criterion,
                        'mean': mean_value,
                        'std': std_value
                    })
                else:
                    print(f"Criterion '{criterion}' not found for year '{year}'. Skipping.")

        # Create a DataFrame from the results
        result_df = pd.DataFrame(results)

        # Visualization: One plot for each criterion
        for criterion in criteria:
            # Filter data for the current criterion
            criterion_data = result_df[result_df['criterion'] == criterion]

            # Plot
            plt.figure(figsize=(8, 5))
            plt.bar(criterion_data['year'].astype(str), criterion_data['mean'], yerr=criterion_data['std'], capsize=5)
            plt.title(f"Mean and SD of {criterion} by Year")
            plt.ylabel("Mean Value")
            plt.xlabel("Year")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        print()


    def merge_cv_results_across_reps(self):
        """
        This function loads the cv_results obtained for individual reps and creates a new file that contains all reps.
        The result of this function should be equivalent to the cluster results if we did not split the reps on the
        cluster. Thus, a .json file in this format:
        {
            "rep_0": {
                "outer_fold_0": {
                    "imp_0": {
                        "r2": 0.3218946246385005,
                        "neg_mean_squared_error": -0.2943621131072718,
                        "spearman": 0.5529809570410639
                    }
                }
        We stored the results in the same folder under "cv_results.json"
        """
        # Traverse the base directory
        for root, dirs, files in os.walk(self.base_result_dir):
            # Initialize an empty dictionary to hold results for the current subdirectory
            all_results = {}

            # Iterate through the files in the current directory
            for file_name in files:
                if file_name.startswith("cv_results_rep_") and file_name.endswith(".json"):
                    # Build the full path to the file
                    file_path = os.path.join(root, file_name)
                    # Load the JSON file
                    with open(file_path, 'r') as f:
                        rep_results = json.load(f)
                    # Extract the rep key from the file name (e.g., "rep_0" from "cv_results_rep_0.json")
                    rep_key = file_name.replace("cv_results_", "").replace(".json", "")
                    # Add to the all_results dictionary
                    all_results[rep_key] = rep_results

            # If results were found, save them in "cv_results.json" in the current subdirectory
            if all_results:
                output_file = os.path.join(root, "cv_results.json")
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=4)

                print(f"Merged cv_results saved to {output_file}")

    def merge_lin_model_coefs_across_reps(self):
        """
        This function loads the best_models_rep_*.pkl files from subdirectories, extracts coefficients for each repetition,
        and stores the merged results as JSON in the corresponding subdirectory.

        The resulting file should look like this:

        {
            "rep_0": {
                "outer_fold_0": {
                    "imputation_0": {
                        "srmc_number_responses": x,
                        "srmc_percentage_responses": y,
        """

        # Traverse the base directory
        for root, dirs, files in os.walk(self.base_result_dir):
            if os.path.basename(root) == "randomforestregressor":
                continue
            # Initialize a dictionary to store coefficients for the current subdirectory
            coefs_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            # Iterate through files in the current directory
            for file_name in files:
                if file_name.startswith("best_models_rep_") and file_name.endswith(".pkl"):
                    # Build the full path to the file
                    file_path = os.path.join(root, file_name)

                    # Extract the rep key from the file name (e.g., "rep_0" from "best_models_rep_0.pkl")
                    rep_key = file_name.replace("best_models_", "").replace(".pkl", "")

                    # Load the pickle file
                    with open(file_path, "rb") as f:
                        best_models_rep = pickle.load(f)

                    # Process each outer fold and imputation
                    for outer_fold_idx, outer_fold in enumerate(best_models_rep):
                        for imputation_idx, model in enumerate(outer_fold):
                            coefs_sub_dict = dict(zip(model.feature_names_in_, model.coef_))

                            # Sort coefficients by absolute value (optional)
                            sorted_coefs_sub_dict = dict(
                                sorted(coefs_sub_dict.items(), key=lambda item: abs(item[1]), reverse=True)
                            )

                            # Store in the dictionary
                            coefs_dict[rep_key][f"outer_fold_{outer_fold_idx}"][f"imputation_{imputation_idx}"] = sorted_coefs_sub_dict

            # If coefficients were collected, save them in a JSON file in the current subdirectory
            if coefs_dict:
                # Convert defaultdict to regular dict for JSON serialization
                regular_dict = self.defaultdict_to_dict(coefs_dict)

                # Save to "lin_model_coefs.json" in the current subdirectory
                output_file = os.path.join(root, "lin_model_coefficients.json")
                with open(output_file, 'w') as f:
                    json.dump(regular_dict, f, indent=4)

                print(f"Merged linear model coefficients saved to {output_file}")

    def merge_shap_values_across_reps(self):
        """
        Merge shap_values_rep_*.pkl files into a single file structured like shap_values.pkl.
        The merged file includes the combined shap_values, base_values, and data for all reps.
        The resulting file should have the following structure
        {
           "shap_values":
                {
                "rep_0": ndarray[n_samples, n_features, n_imputations],
            "base_values": ...

        """
        # TODO Improve
        test_dir = "../results/res_1111_oh_shap/pl_mac/selected/state_wb/randomforestregressor/shap_values.pkl"
        # Load the pickle file
        with open(test_dir, "rb") as f:
            shap_example = pickle.load(f)

        # Initialize the merged dictionary
        merged_shap_values = {
            "shap_values": defaultdict(dict),
            "base_values": defaultdict(dict),
            "data": defaultdict(dict),
        }

        # Traverse the base directory for shap_values_rep_*.pkl files
        for root, dirs, files in os.walk(self.base_result_dir):
            for file_name in files:
                if file_name.startswith("shap_values_rep_") and file_name.endswith(".pkl"):
                    # Build the full path to the file
                    file_path = os.path.join(root, file_name)

                    # Extract the repetition key (e.g., "rep_0" from "shap_values_rep_0.pkl")
                    rep_key = file_name.replace("shap_values_", "").replace(".pkl", "")

                    # Load the pickle file
                    with open(file_path, "rb") as f:
                        shap_example = pickle.load(f)

                    # Merge shap_values, base_values, and data for the current repetition
                    merged_shap_values["shap_values"][rep_key] = shap_example["shap_values"]
                    merged_shap_values["base_values"][rep_key] = shap_example["base_values"]
                    merged_shap_values["data"][rep_key] = shap_example["data"]
                    # TODO Add feature names

        # Convert defaultdict to dict for serialization
        merged_shap_values["shap_values"] = dict(merged_shap_values["shap_values"])
        merged_shap_values["base_values"] = dict(merged_shap_values["base_values"])
        merged_shap_values["data"] = dict(merged_shap_values["data"])

        # Save the merged shap_values.pkl file
        output_file = os.path.join(self.base_result_dir, "shap_values.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(merged_shap_values, f)

        print(f"Merged shap_values saved to {output_file}")

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
            results_file_name = f'{self.result_filenames[result_type]}.json'
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
            results_proc = f"proc_{results_raw_filename}.json"  # JSON file extension

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

    def defaultdict_to_dict(self, dct):
        if isinstance(dct, defaultdict):
            dct = {k: self.defaultdict_to_dict(v) for k, v in dct.items()}
        return dct




