import os
import pickle
import random
import string
from collections import defaultdict

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import shutil

from src.utils.DataLoader import DataLoader


class ShapProcessor:
    """
    This class processes the SHAP values. It
        - summarizes the SHAP values obtained across outer folds
        - recreates the SHAP explanation objects for plotting
    """

    def __init__(self, var_cfg, base_result_dir, processed_output_path, name_mapping):
        self.var_cfg = var_cfg
        self.name_mapping = name_mapping
        self.base_result_dir = base_result_dir
        self.processed_output_path = processed_output_path
        self.data_loader = DataLoader()

        self.data_importance_plot = None
        self.data_violin_plot = None

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

    def prepare_data(self, model_to_plot: str, crit_to_plot: str, samples_to_include: str, custom_affordances: dict = None):
        """
        Prepares the data for SHAP visualization. Allows using custom sample inclusion
        values for specific feature combinations as specified in custom_affordances.

        Args:
            model_to_plot (str): The model name to filter by.
            crit_to_plot (str): The criterion name to filter by.
            samples_to_include (str): Default samples to include.
            custom_affordances (dict): Dictionary with feature combinations as keys and
                                       custom samples_to_include values as values.
                                       e.g., {"sens": "selected", "mac": "selected"}
        Returns:
            dict: Nested dictionary with SHAP explanation objects.
        """
        # Initialize the root result dictionary with nested defaultdicts
        result_dct = self.nested_dict()

        # Traverse the directory structure
        for root, dirs, files in os.walk(self.processed_output_path):
            if 'shap_values_summary.pkl' in files:  # 'shap_values_processed.pkl'
                # Extract the directory components based on the structure
                relative_path = os.path.relpath(root, self.processed_output_path)
                parts = relative_path.split(os.sep)

                # Reorder parts from a/b/c/d to c/b/d/a and check conditions
                if len(parts) == 4:
                    predictor_combination, samples, crit, model = parts

                    # Possibility to mix up
                    if custom_affordances:
                        if predictor_combination in custom_affordances:
                            effective_samples_to_include = custom_affordances[predictor_combination]
                        else:
                            effective_samples_to_include = samples_to_include
                    else:
                        effective_samples_to_include = samples_to_include

                    # Only load if all filters match
                    if crit == crit_to_plot and samples == effective_samples_to_include and model == model_to_plot:
                        # Get the path to the shap_values file
                        shap_values_path = os.path.join(root, 'shap_values_summary.pkl')
                        # Load the shap_values using the data loader
                        shap_values = self.data_loader.read_pkl(shap_values_path)

                        # Apply name mapping
                        feature_names = shap_values["feature_names"]
                        # print(feature_names)
                        filtered_feature_names = [feature for feature in feature_names if feature not in ['other_unique_id', 'other_country', 'other_years_of_participation']]
                        # TODO sauberer klären
                        #feature_names = [name for name in feature_names if name.startswith("mac")] + [name for name in feature_names if not name.startswith("mac")]
                        #feature_names = self.apply_name_mapping(feature_names)

                        # Recreate explanation objects
                        shap_exp = self.recreate_shap_exp_objects(
                            shap_values=np.array(shap_values["shap_values"]["mean"]),
                            base_values=np.array(shap_values["base_values"]["mean"]),
                            data=np.array(shap_values["data"]["mean"]),
                            feature_names=filtered_feature_names,
                        )
                        # Store explanation objects in the nested defaultdict structure
                        # result_dct[crit][samples][model][predictor_combination] = shap_exp
                        result_dct[predictor_combination] = shap_exp
        return result_dct
        # Convert nested defaultdict structure to a regular dictionary
        # self.data_importance_plot = self.defaultdict_to_dict(result_dct)

        # self.data_violin_plot = self.defaultdict_to_dict(result_dct)

    def apply_name_mapping(self, features: list) -> list:
        """
        Maps a list of features to their descriptive names based on the second-level name in the provided mapping dictionary.

        Args:
            features (list): A list of feature strings in the format "first_level_second_level".
            mapping_dict (dict): A dictionary containing mappings for the second hierarchy level.

        Returns:
            list: A list of mapped second-level feature names.
        """
        mapped_features = []

        for feature in features:
            # Split into first and second parts based on the first underscore
            first_level, _, second_level = feature.partition('_')

            # Map only the second level, if available
            second_level_mapped = self.name_mapping.get(first_level, {}).get(second_level, second_level)

            # Append only the mapped second level
            mapped_features.append(second_level_mapped)

        return mapped_features



    @staticmethod
    def recreate_shap_exp_objects(shap_values, base_values, data, feature_names):
        """

        Args:
            shap_values:
            base_values:
            data:
            features:

        Returns:

        """
        explanation = shap.Explanation(values=shap_values, base_values=base_values, data=data, feature_names=feature_names)
        return explanation

    def defaultdict_to_dict(self, d):
        """Recursively converts a defaultdict to a standard dict."""
        if isinstance(d, defaultdict):
            d = {k: self.defaultdict_to_dict(v) for k, v in d.items()}
        return d

    @classmethod
    def nested_dict(cls):
        """Creates a nested defaultdict that can be used to create an arbitrary depth dictionary."""
        return defaultdict(cls.nested_dict)

    @staticmethod
    def compute_mean_sd(shap_value_dct):
        """

        Args:
            shap_value_dct:

        Returns:


        """
        # Initialize results dictionary
        results = {}

        # Iterate over the keys of interest
        for key in ["shap_values", "data", "base_values"]:
            # Collect the arrays across repetitions
            values = np.array([shap_value_dct[key][f'rep_{i}'] for i in range(10)])  # TODO set in config, not hardcode

            # Calculate mean and standard deviation across repetitions and imputations
            if key in ["shap_values", "data"]:
                imp_axis = 3
            else:  # Base Values have no third dimension
                imp_axis = 2
            results[key] = {
                "mean": np.mean(values, axis=(0, imp_axis)),
                "std": np.std(values, axis=(0, imp_axis))
            }
        results["feature_names"] = shap_value_dct["feature_names"]

        return results

    @staticmethod
    def process_feature_names(feature_names, rel_path, feature_mappings):
        """
        This function processes the feature names. It
            - removes the meta-columns
            - corrects the order in case of country-level, individual-level combinations

        Args:
            feature_names (list):
            rel_path (str):

        Returns:
            feature_names_processed (list):
        """
        meta_vars = [col for col in feature_names if col.startswith("other")]
        feature_names_processed = [col for col in feature_names if col not in meta_vars]

        if "_mac" in rel_path:
            #feature_names_processed
            #cols = df.columns[3:]
            for feat_combo, mapping in feature_mappings.items():
                if len(feature_names_processed) == len(mapping):
                    feature_names_processed = [mapping.get(col, col) for col in feature_names_processed]
                else:
                    continue
        return feature_names_processed

        #if "mac" in rel_path:
        #    country_feature_lst = [col for col in feature_names_processed if col.startswith("mac")]
        #    individual_feature_lst = [col for col in feature_names_processed if not col.startswith("mac")]
        #    feature_names_processed = individual_feature_lst + country_feature_lst

    def recreate_explanation_objects(self):
        """
        This function recreates the SHAP explanation objects from the
            - SHAP value arrays
            - Base value arrays
            - Features
        Returns:
        """
        # TODO This is more of a test script if we can succefully rebuild the Explanation objects -> Adjust
        res_dir = "../results/cluster_results/pl_srmc_mac/control/state_wb/randomforestregressor"
        filename = os.path.join(res_dir, "shap_values.pkl")
        shap_results = self.data_loader.read_pkl(filename)

        # Make example for rep0, average across imputations
        shap_vals = shap_results['shap_values']["rep_0"]
        shap_vals = np.mean(shap_vals, axis=2)
        base_vals = shap_results['base_values']["rep_0"]
        base_vals = np.mean(base_vals, axis=1).flatten()
        shap_data = shap_results['data']["rep_0"]
        shap_data = np.mean(shap_data, axis=2)
        features = shap_results["feature_names"]

        features.remove("other_unique_id")
        features.remove("other_country")
        features.remove("other_years_of_participation")
        shap_df = pd.DataFrame(shap_data, columns=features)
        explanation_test = shap.Explanation(shap_vals, base_vals, data=shap_data, feature_names=features)

        shap.summary_plot(explanation_test.values, shap_df)
        shap.plots.waterfall(explanation_test[10])
        shap.plots.violin(explanation_test)
        plt.show()
        print("####")
