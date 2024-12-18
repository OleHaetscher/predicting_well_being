import os
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from shap import Explanation

from src.utils.DataLoader import DataLoader
from src.utils.utilfuncs import apply_name_mapping, defaultdict_to_dict
from mpl_toolkits.mplot3d import Axes3D

class ShapProcessor:
    """
    This class processes the SHAP values. It
        - summarizes the SHAP values obtained across outer folds
        - recreates the SHAP explanation objects for plotting
    """

    def __init__(self, var_cfg, processed_output_path, name_mapping):
        self.var_cfg = var_cfg
        self.name_mapping = name_mapping
        # self.base_result_dir = base_result_dir
        self.processed_output_path = processed_output_path
        self.shap_ia_values_path = self.var_cfg["postprocessing"]["shap_ia_values_path"]
        self.data_loader = DataLoader()
        self.shap_values_file_name = self.var_cfg["postprocessing"]["summarized_file_names"]["shap_values"]
        self.shap_ia_values_file_name = self.var_cfg["postprocessing"]["summarized_file_names"]["shap_ia_values"]
        self.meta_vars = ['other_unique_id', 'other_country', 'other_years_of_participation']

    @classmethod
    def nested_dict(cls):
        """Creates a nested defaultdict that can be used to create an arbitrary depth dictionary."""
        return defaultdict(cls.nested_dict)

    def prepare_shap_data(self, model_to_plot: str, crit_to_plot: str, samples_to_include: str, col_assignment: list[list]) -> dict:
        """
        Prepares the data for SHAP visualization. Allows using custom sample inclusion
        values for specific feature combinations as specified in custom_affordances.

        Args:
            model_to_plot (str): The model name to filter by.
            crit_to_plot (str): The criterion name to filter by.
            samples_to_include (str): Default samples to include.
            col_assignment: list: Defined the position of a feature combination in the plot

        Returns:
            dict: Nested dictionary with SHAP explanation objects.
        """
        # Initialize the root result dictionary with nested defaultdicts
        result_dct = self.nested_dict()

        # Traverse the directory structure
        for root, dirs, files in os.walk(self.processed_output_path):
            if self.shap_values_file_name in files:   # 'shap_values_summary.pkl'

                # Extract the directory components based on the structure
                relative_path = os.path.relpath(root, self.processed_output_path)
                parts = relative_path.split(os.sep)

                # Reorder parts and check conditions
                if len(parts) == 4:
                    feature_combination, samples, crit, model = parts

                    # Determine which samples filter to use
                    if samples_to_include == "combo":
                        required_samples = self.get_required_sample_for_combo(
                            feature_combination=feature_combination,
                            col_assignment=col_assignment,
                        )
                    else:
                        required_samples = samples_to_include

                    # Make better
                    if feature_combination == "all_in":
                        required_samples = "all"

                    # Only load if all filters match
                    if crit == crit_to_plot and samples == required_samples and model == model_to_plot:
                        shap_values_path = os.path.join(root, self.shap_values_file_name)
                        shap_values = self.data_loader.read_pkl(shap_values_path)

                        # Filter out meta vars from feature names
                        feature_names = [feature for feature in shap_values["feature_names"]
                                         if feature not in self.meta_vars]
                        feature_names = apply_name_mapping(
                            features=feature_names,
                            name_mapping=self.name_mapping,
                            prefix=True
                        )

                        # Recreate explanation objects
                        shap_exp = self.recreate_shap_exp_objects(
                            shap_values=np.array(shap_values["shap_values"]["mean"]),
                            base_values=np.array(shap_values["base_values"]["mean"]),
                            data=np.array(shap_values["data"]["mean"]),
                            feature_names=feature_names,
                        )
                        result_dct[feature_combination] = shap_exp
        return result_dct

    def prepare_shap_ia_data(self,
                             model_to_plot: str,
                             crit_to_plot: str,
                             samples_to_include: str,
                             feature_combination_to_plot: str,
                             meta_stat_to_extract: str,
                             stat_to_extract: str,
                             order_to_extract: int,
                             num_to_extract: int) -> dict:
        """
        Prepares the data for SHAP visualization. Allows using custom sample inclusion
        values for specific feature combinations as specified in custom_affordances.

        Args:
            model_to_plot (str): The model name to filter by.
            crit_to_plot (str): The criterion name to filter by.
            samples_to_include (str): Default samples to include.
            col_assignment: list: Defined the position of a feature combination in the plot

        Returns:
            dict: Nested dictionary with SHAP explanation objects.
        """
        result_dct = self.nested_dict()

        for root, dirs, files in os.walk(self.shap_ia_values_path):
            if str(self.shap_ia_values_file_name) in files:   # 'shap_ia_values_summary.pkl'
                relative_path = os.path.relpath(root, self.shap_ia_values_path)
                parts = relative_path.split(os.sep)

                if len(parts) == 4:
                    feature_combination, samples, crit, model = parts

                    if (crit == crit_to_plot and samples == samples_to_include and
                            model == model_to_plot and feature_combination == feature_combination_to_plot):

                        shap_ia_values_path = os.path.join(str(root), str(self.shap_ia_values_file_name))
                        shap_ia_values = self.data_loader.read_pkl(shap_ia_values_path)

                        # TODO Processing the samples values, not the aggregates!!
                        # base_values = np.array(shap_ia_values["base_values_sample"]["mean"])
                        base_value = self.get_base_values(root)

                        shap_ia_values_dct = {
                            key: value['mean']
                            for key, value
                            in shap_ia_values["top_interactions"]["top_abs_interactions_of_sample"].items()
                            if isinstance(key, tuple) and len(key) > 1
                        }
                        # Sort the dictionary by absolute mean values of NumPy arrays
                        shap_ia_values_dct = dict(sorted(
                            shap_ia_values_dct.items(),
                            key=lambda item: abs(np.mean(item[1])),
                            reverse=True  # Optional: Sort in descending order
                        ))
                        feature_tuples = list(shap_ia_values_dct.keys())

                        # Map the keys using the name_mapping function
                        formatted_features_dct = {}
                        for feature_pair, values in shap_ia_values_dct.items():
                            # Get x seperated string of the two interacting variables
                            mapped_key = " x ".join(apply_name_mapping([k], self.name_mapping, prefix=True)[0] for k in feature_pair)
                            formatted_features_dct[mapped_key] = values

                        shap_ia_values_arr = np.array([
                            value for value in formatted_features_dct.values()
                        ])
                        feature_names = list(formatted_features_dct.keys())

                        # get data
                        data = self.get_ia_feature_data(root_path=root, top_n_interactions=feature_tuples)

                        # Recreate the explanation object (pretending this were ordinary shap values)
                        shap_ia_exp = self.recreate_shap_exp_objects(
                            shap_values=shap_ia_values_arr.T,
                            base_values=base_value,
                            feature_names=feature_names,
                            data=data.values
                        )

                        result_dct[f"{feature_combination}_ia_values"] = shap_ia_exp

        return defaultdict_to_dict(result_dct)

    def get_ia_feature_data(self, root_path, top_n_interactions: list[tuple[str]]):
        """
        This method loads the feature values corresponding to the current SHAP interaction analysis and
        takes the mean of both features to display the SHAP beeswarm plots

        Args:
            root_path (str): Path to load the data
            top_n_interactions (list[tuple[str]]): List of tuples containing the n strongest abs order 2 interactions

        Returns:
            pd.DataFrame: DataFrame with new columns representing the mean of feature pairs from top_n_interactions
        """
        # Load SHAP values file
        file_name = os.path.join(root_path, self.shap_values_file_name)
        shap_values = self.data_loader.read_pkl(file_name)

        # Extract relevant data
        data = shap_values["data"]["mean"]
        feature_names = shap_values["feature_names"][3:]

        # Create a DataFrame from the data
        feature_df = pd.DataFrame(data, columns=feature_names)

        # Add new columns based on mean of the specified feature pairs in top_n_interactions
        for interaction in top_n_interactions:
            if len(interaction) != 2:
                raise ValueError(f"Each interaction must contain exactly two feature names. Invalid entry: {interaction}")

            feature1, feature2 = interaction
            if feature1 not in feature_df.columns or feature2 not in feature_df.columns:
                raise ValueError(f"Features {feature1} and {feature2} must be present in the DataFrame columns.")

            # Create a new column name for the interaction
            interaction_col_name = (feature1, feature2)

            # Calculate the mean of the two features and add as a new column
            feature_df[interaction_col_name] = feature_df[[feature1, feature2]].mean(axis=1)

        feature_df = feature_df.drop(columns=feature_names)

        return feature_df

    def get_base_values(self, root_path):
        """

        Args:
            root_path:

        Returns:

        """
        # Load SHAP values file
        file_name = os.path.join(root_path, self.shap_values_file_name)
        shap_values = self.data_loader.read_pkl(file_name)

        # Extract relevant data
        base_values = shap_values["base_values"]["mean"]
        return base_values

    @staticmethod
    def get_required_sample_for_combo(feature_combination: str, col_assignment: list[list]):
        """
        For the plot in the paper, we need to adjust "samples_to_include" based on the specific feature combination.
        Therefore, we need a custom mapping to load the right data

        Args:
            feature_combination: str, the current feature_combination
            col_assignment: list, the feature_combination and its associated location in the plot

        Returns:
            str: samples_to_include for the current feature_combination
        """
        # If combination is in the first sublist, use "selected"
        if feature_combination in col_assignment["first_col"]:
            return "selected"
        # If combination is in the second or third sublist, use "all"
        for sublst in col_assignment["second_col"] + col_assignment["third_col"]:
            if feature_combination in sublst:
                return "all"
        # raise ValueError(f"Predictor combination {feature_combination} not found in col_assignment.")

    @staticmethod
    def recreate_shap_exp_objects(
            shap_values: np.ndarray,
            base_values: np.ndarray,
            feature_names: list = None,
            data: np.ndarray = None,
    ) -> Explanation:
        """
        This method recreates the SHAP explanation objects for more flexibility when plotting the data.

        Args:
            shap_values: 2darray, containing the shap_values
            base_values: 1darray, containing the base_values
            data: 2darray, containing the data
            feature_names: list, containing the feature names corresponding to the shap values

        Returns:
            SHAP.Explanation object
        """
        explanation = shap.Explanation(values=shap_values, base_values=base_values, data=data, feature_names=feature_names)
        return explanation

    def defaultdict_to_dict(self, dct: Union[defaultdict, dict]) -> dict:
        """
        Recursively converts a defaultdict to a standard dict.

        Args:
            dct: A defaultdict to convert or a dict (will remain unchanged)

        Returns:
            dict
        """
        if isinstance(dct, defaultdict):
            dct = {k: self.defaultdict_to_dict(v) for k, v in dct.items()}
        return dct
