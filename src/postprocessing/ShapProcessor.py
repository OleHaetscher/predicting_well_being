import os
from collections import defaultdict
from typing import Union

import numpy as np
import shap
from shap import Explanation

from src.utils.DataLoader import DataLoader
from src.utils.utilfuncs import apply_name_mapping, defaultdict_to_dict


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
                             feature_combination: str,
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
            print(files)
            if str(self.shap_ia_values_file_name) in files:   # 'shap_ia_values_summary.pkl'
                relative_path = os.path.relpath(root, self.shap_ia_values_path)
                parts = relative_path.split(os.sep)

                if len(parts) == 4:
                    feature_combination, samples, crit, model = parts

                    if crit == crit_to_plot and samples == samples_to_include and model == model_to_plot:
                        shap_ia_values_path = os.path.join(str(root), str(self.shap_ia_values_file_name))
                        shap_ia_values = self.data_loader.read_pkl(shap_ia_values_path)

                        base_values = np.array(shap_ia_values["base_values_sample"]["mean"])
                        # feature pairs as keys, shap ia values as values
                        shap_ia_values_dct = {
                            key: value['mean']
                            for key, value in shap_ia_values['ia_values_sample'].items()
                            if isinstance(key, tuple) and len(key) > 1
                        }

                        # Map the keys using the name_mapping function
                        formatted_features_dct = {}
                        for feature_pair, values in shap_ia_values_dct.items():
                            # Get x seperated string of the two interacting variables
                            mapped_key = " x ".join(apply_name_mapping([k], self.name_mapping, prefix=True)[0] for k in feature_pair)
                            formatted_features_dct[mapped_key] = values

                        shap_ia_values_arr = np.array([
                            value for value in formatted_features_dct.values()
                        ])
                        data = np.random.rand(*shap_ia_values_arr.shape)

                        # Recreate the explanation object (pretending this were ordinary shap values)
                        shap_ia_exp = self.recreate_shap_exp_objects(
                            shap_values=shap_ia_values_arr,
                            base_values=base_values,
                            feature_names=list(formatted_features_dct.keys()),
                            data=data  # TODO: We may replace this with the actual data
                        )

                        result_dct[f"{feature_combination}_ia_values"] = shap_ia_exp

        return defaultdict_to_dict(result_dct)

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
            feature_names: list,
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
