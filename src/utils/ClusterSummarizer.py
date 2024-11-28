import itertools
import json
import os
import pickle
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


class ClusterSummarizer:
    """
    Class that summarizes results on the cluster to
        - compute M and SD of the cv results
        - Aggregate SHAP values across repetitions
    """
    def __init__(self, base_result_dir, var_config_path="../../configs/config_var.yaml", num_reps=10):
        """
        Constructor, needs a directory
        """
        self.base_result_dir = base_result_dir
        with open(var_config_path, "r") as f:
            var_cfg = yaml.safe_load(f)
        self.var_cfg = var_cfg
        self.num_reps = num_reps
        self.rng_ = np.random.RandomState(self.var_cfg["analysis"]["random_state"])  # Local RNG

    def aggregate(self):
        """
        This method walks through the directory tree from self.base_result_dir and
            - checks if a filetype (cv_results / lin_model_coefs / shap_values) exists for all repetitions
            - aggregates the contents across repetitions and imputations
            - saves the aggregates to a file
        """
        for root, dirs, files in os.walk(self.base_result_dir):
            print("---")
            print(root, dirs)
            # Summarize cv_results
            if all(f"cv_results_rep_{i}.json" in files for i in range(self.num_reps)):
                cv_results_files = [f"cv_results_rep_{i}.json" for i in range(self.num_reps)]
                print(f"process cv_results in {root}")
                summary = self.summarize_cv_results(root, cv_results_files)
                with open(os.path.join(root, "cv_results_summary.json"), "w") as f:
                    json.dump(summary, f, indent=4)
                print(f"stored summarized cv_results in {root}")

            # Summarize linear model coefficients
            if all(f"best_models_rep_{i}.pkl" in files for i in range(self.num_reps)):
                if "elasticnet" in root:
                    best_models_files = [f"best_models_rep_{i}.pkl" for i in range(self.num_reps)]
                    print(f"process lin_model_coefs in {root}")
                    summary = self.summarize_lin_model_coefs(root, best_models_files)
                    with open(os.path.join(root, "lin_model_coefs_summary.json"), "w") as f:
                        json.dump(summary, f, indent=4)
                    print(f"stored summarized lin_model_coefs in {root}")

            # Summarize SHAP values
            if all(f"shap_values_rep_{i}.pkl" in files for i in range(self.num_reps)):
                shap_files = [f"shap_values_rep_{i}.pkl" for i in range(self.num_reps)]
                print(f"process SHAP values in {root}")
                summary = self.summarize_shap_values(root, shap_files)
                with open(os.path.join(root, "shap_values_summary.pkl"), "wb") as f:
                    pickle.dump(summary, f)
                print(f"stored summarized shap_values in {root}")

            # Summarize SHAP IA Values
            if all(f"shap_ia_values_rep_{i}.pkl" in files for i in range(self.num_reps)):
                shap_ia_values_files = [f"shap_ia_values_rep_{i}.pkl" for i in range(self.num_reps)]
                shap_ia_base_values_files = [f"shap_ia_base_values_rep_{i}.pkl" for i in range(self.num_reps)]
                print(f"process SHAP IA values in {root}")
                summary = self.summarize_shap_ia_values(root, shap_ia_values_files, shap_ia_base_values_files)
                with open(os.path.join(root, "shap_ia_values_summary.pkl"), "wb") as f:
                    pickle.dump(summary, f)
                print(f"stored summarized shap_ia_values in {root}")
    def summarize_metrics(self, data_records, identifier_cols):
        """
        Summarizes metrics from data_records.

        Parameters:
        - data_records: List of dictionaries containing the data.
        - identifier_cols: List of columns that are identifiers (e.g., ['rep', 'outer_fold', 'imputation']).

        Returns:
        - summary: Dictionary containing the summary statistics.
        """
        df = pd.DataFrame(data_records)

        # Identify metric columns
        metric_cols = [col for col in df.columns if col not in identifier_cols]

        # Compute overall mean and std across all data points
        overall_mean = df[metric_cols].mean().round(4).to_dict()
        overall_std = df[metric_cols].std(ddof=0).round(5).to_dict()

        # Compute sd within folds across imputations
        sd_within_folds = df.groupby(['rep', 'outer_fold'])[metric_cols].std(ddof=0)
        sd_within_folds_across_imps = sd_within_folds.mean().round(5).to_dict()

        # Compute sd across folds within imputations
        sd_across_folds = df.groupby(['rep', 'imputation'])[metric_cols].std(ddof=0)
        sd_across_folds_within_imps = sd_across_folds.mean().round(5).to_dict()

        # Compile summary
        summary = {
            'm': overall_mean,
            'sd_across_folds_imps': overall_std,
            'sd_within_folds_across_imps': sd_within_folds_across_imps,
            'sd_across_folds_within_imps': sd_across_folds_within_imps
        }

        return summary

    def summarize_cv_results(self, result_dir, file_names):
        """
        Summarizes CV results from multiple files (reps) and their hierarchical structure
        (outer folds and imputations).
        """
        # Collect all metric records into a list
        data_records = []

        for rep_idx, file_name in enumerate(file_names):
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)

            for outer_fold_key, imps in data.items():
                outer_fold_idx = int(outer_fold_key.split('_')[-1])
                for imp_key, metrics in imps.items():
                    imp_idx = int(imp_key.split('_')[-1])
                    record = {
                        'rep': rep_idx,
                        'outer_fold': outer_fold_idx,
                        'imputation': imp_idx
                    }
                    record.update(metrics)
                    data_records.append(record)

        # Define identifier columns
        identifier_cols = ['rep', 'outer_fold', 'imputation']

        # Call the common summarization method
        summary = self.summarize_metrics(data_records, identifier_cols)

        return summary

    def summarize_lin_model_coefs(self, result_dir, file_names):
        """
        Aggregates linear model coefficients across repetitions, outer folds, and imputations.
        Computes mean coefficients and various standard deviations.
        Handles cases where models have different subsets of features due to feature selection.
        """
        import pandas as pd
        import numpy as np
        import os
        import pickle
        from collections import defaultdict

        data_records = []

        for rep_idx, file_name in enumerate(file_names):
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)  # Data is a nested list: outer folds -> imputations

            for outer_fold_idx, imps in enumerate(data):  # Iterate over outer folds
                for imp_idx, model in enumerate(imps):  # Iterate over imputations
                    if hasattr(model, 'coef_'):
                        # Use feature names and coefficients
                        coefs = dict(zip(model.feature_names_in_, model.coef_.ravel()))
                        record = {
                            'rep': rep_idx,
                            'outer_fold': outer_fold_idx,
                            'imputation': imp_idx
                        }
                        # Add coefficients to the record
                        for feature_name, coef_value in coefs.items():
                            record[feature_name] = coef_value
                        data_records.append(record)

        # Define identifier columns
        identifier_cols = ['rep', 'outer_fold', 'imputation']

        # Call summarize_metrics to compute m and sd values
        summary = self.summarize_metrics(data_records, identifier_cols)

        # Convert data_records to DataFrame to compute non-zero counts
        df = pd.DataFrame(data_records)

        # Identify metric columns (coefficients)
        metric_cols = [col for col in df.columns if col not in identifier_cols]

        # Compute non-zero coefficient counts
        non_zero_counts = (df[metric_cols] != 0).sum().to_dict()

        # Sort features based on the absolute value of the mean coefficients
        sorted_features = sorted(summary['m'].keys(), key=lambda k: abs(summary['m'][k]), reverse=True)

        # Reorder dictionaries in summary based on sorted features
        summary['m'] = {k: summary['m'][k] for k in sorted_features}
        summary['sd_across_folds_imps'] = {k: summary['sd_across_folds_imps'][k] for k in sorted_features}
        summary['sd_within_folds_across_imps'] = {k: summary['sd_within_folds_across_imps'][k] for k in sorted_features}
        summary['sd_across_folds_within_imps'] = {k: summary['sd_across_folds_within_imps'][k] for k in sorted_features}
        summary['non_zero_coefs'] = {k: non_zero_counts.get(k, 0) for k in sorted_features}

        return summary

    def summarize_shap_values(self, result_dir, file_names):
        """
        Aggregates SHAP values across repetitions and imputations.
        """
        # Initialize storage for SHAP data
        shap_values_list = []
        data_list = []
        base_values_list = []
        feature_names = None

        # Iterate over files and load SHAP data
        for file_name in file_names:
            file_path = os.path.join(result_dir, file_name)
            with open(file_path, 'rb') as f:
                shap_data = pickle.load(f)

            # Simplified appending of data
            if "shap_values" in shap_data:
                shap_values_list.append(np.expand_dims(shap_data["shap_values"], axis=0))  # Add repetition dimension
            if "data" in shap_data:
                data_list.append(np.expand_dims(shap_data["data"], axis=0))  # Add repetition dimension
            if "base_values" in shap_data:
                base_values_list.append(np.expand_dims(shap_data["base_values"], axis=0))  # Add repetition dimension

            if feature_names is None:
                feature_names = shap_data.get("feature_names")

        # Concatenate along the repetition axis
        shap_values_array = np.concatenate(shap_values_list, axis=0)  # Shape: (reps, ...)
        data_array = np.concatenate(data_list, axis=0)  # Shape: (reps, ...)
        base_values_array = np.concatenate(base_values_list, axis=0)  # Shape: (reps, ...)

        # Compute mean and standard deviation
        results = {}
        for key, values in zip(["shap_values", "data", "base_values"],
                               [shap_values_array, data_array, base_values_array]):
            # Determine imputation axis (3 if there's an additional dimension, otherwise 2)
            imp_axis = 3 if values.ndim > 3 else 2

            # Compute mean and standard deviation across repetitions and imputations
            results[key] = {
                "mean": np.mean(values, axis=(0, imp_axis)).tolist(),  # Average across repetitions and imputations
                "std": np.std(values, axis=(0, imp_axis)).tolist()  # Std dev across repetitions and imputations
            }

        # Add feature names
        results["feature_names"] = feature_names

        return results

    def summarize_shap_ia_values(self, result_dir, shap_ia_values_files, shap_ia_base_values_files):
        # Initialize dictionaries to collect results across repetitions
        ia_values_all_reps = {}
        base_values_all_reps = {}

        # load ia_value_mappings from current folder -> this has to be the same for all reps
        mapping_path = os.path.join(result_dir, "ia_values_mappings_rep_0.pkl")
        with open(mapping_path, "rb") as file:
            mapping_dct = pickle.load(file)
            combo_index_mapping = mapping_dct["combo_index_mapping"]
            feature_index_mapping = mapping_dct["feature_index_mapping"]
            num_combos = mapping_dct["num_combos"]

        # Loop over all repetitions for SHAP IA values
        for rep_idx, file_name in enumerate(shap_ia_values_files):
            file_path = os.path.join(result_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    shap_ia_results_reps_imps = pickle.load(file)
                ia_values_all_reps[f"rep_{rep_idx}"] = shap_ia_results_reps_imps
            else:
                print(f"Warning: SHAP IA values file for repetition {rep_idx} not found.")
                continue

        # Loop over all repetitions for base values
        for rep_idx, file_name in enumerate(shap_ia_base_values_files):
            file_path = os.path.join(result_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    base_values_reps_imps = pickle.load(file)
                base_values_all_reps[f"rep_{rep_idx}"] = base_values_reps_imps
            else:
                print(f"Warning: SHAP IA base values file for repetition {rep_idx} not found.")
                continue

        # Now proceed with aggregation across repetitions and imputations
        ia_values_agg_reps_imps, base_value_agg_reps_imps = self.agg_ia_values_across_reps(
            ia_value_dct=ia_values_all_reps,
            base_value_dct=base_values_all_reps,
        )

        # Aggregate results across samples
        abs_ia_values_agg_reps_imps_samples, ia_values_agg_reps_imps_samples, \
        abs_base_value_agg_reps_imps_samples, base_value_agg_reps_imps_samples = self.agg_results_across_samples(
            ia_value_dct=ia_values_agg_reps_imps.copy(),
            base_value_dct=base_value_agg_reps_imps.copy(),
        )

        # Map combinations to features - abs ia values across samples
        abs_ia_values_agg_reps_imps_samples = self.map_combo_to_features(
            abs_ia_values_agg_reps_imps_samples,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping
        )
        # Map combinations to features - raw ia values across samples
        ia_values_agg_reps_imps_samples = self.map_combo_to_features(
            ia_values_agg_reps_imps_samples,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping
        )
        # Map combinations to features - ia values not aggregated across samples
        ia_values_agg_reps_imps = self.map_combo_to_features(
            ia_values_agg_reps_imps,
            feature_index_mapping=feature_index_mapping,
            combo_index_mapping=combo_index_mapping,
        )
        # Sample aggregated results
        num_samples = self.var_cfg["analysis"]["shap_ia_values"]["num_samples"]
        ia_values_sample, base_values_sample = self.sample_aggregated_results(
            ia_values_agg_reps_imps,
            base_value_agg_reps_imps,
            num_samples=num_samples
        )

        # Prepare results dictionary to return
        shap_ia_results_processed = {
            "ia_values_sample": ia_values_sample,
            "base_values_sample": base_values_sample
        }

        # Get top interactions # here we want the abs and the raw means
        top_interactions_per_order = self.get_top_interactions(
            abs_mapped_results_agg_samples=abs_ia_values_agg_reps_imps_samples,
            mapped_results_agg_samples=ia_values_agg_reps_imps_samples,
            mapped_results_no_agg_samples=ia_values_agg_reps_imps,

        )

        # Get most interacting features # here we only need the abs means
        top_interacting_features = self.get_most_interacting_features(
            abs_ia_values_agg_reps_imps_samples
        )

        shap_ia_results_processed["top_interactions"] = top_interactions_per_order
        shap_ia_results_processed["top_interacting_features"] = top_interacting_features
        shap_ia_results_processed['abs_ia_value_agg_reps_imps_samples'] = abs_ia_values_agg_reps_imps_samples

        print("Processed all SHAP IA values.")

        return shap_ia_results_processed

    @staticmethod
    def agg_ia_values_across_reps(ia_value_dct, base_value_dct):
        """
        Aggregates the mean and standard deviation of interaction attribution (IA) values across repetitions
        preserving the samples and combinations dimensions.
        NOTE: Aggregation across imps already took place

        Args:
            ia_value_dct (dict): Dictionary where keys are repetitions, and values are ndarrays of shape (samples, combinations, imputations).
            base_value_dct (dict): Dictionary where keys are repetitions, and values are ndarrays of shape (samples, imputations).

        Returns:
            tuple (dict): A dictionary with aggregated mean and standard deviation for IA and base values.
        """

        # Stack the IA values across repetitions into a single ndarray
        ia_values_array = np.stack(list(ia_value_dct.values()), axis=-1)  # Shape: (samples, combinations, imputations, repetitions)

        # Compute mean and std across imputations and repetitions (axes=-2 and -1)
        ia_mean = np.mean(ia_values_array, axis=-1)  # Resulting shape: (samples, combinations)
        ia_std = np.std(ia_values_array, axis=-1)  # Resulting shape: (samples, combinations)

        # Stack the base values across repetitions into a single ndarray
        base_values_array = np.stack(list(base_value_dct.values()), axis=-1)  # Shape: (samples, imputations, repetitions)

        # Compute mean and std across imputations and repetitions (axes=-2 and -1)
        base_mean = np.mean(base_values_array, axis=-1)  # Resulting shape: (samples,)
        base_std = np.std(base_values_array, axis=-1)  # Resulting shape: (samples,)

        return ({'mean': ia_mean, 'std': ia_std},
                {'mean': base_mean, 'std': base_std})

    @staticmethod
    def agg_results_across_samples(ia_value_dct, base_value_dct):
        """
        Aggregates the mean and standard deviation across samples from the output of the previous aggregation function.
        Note: For IA and base values, we compute both the absolute mean and the raw mean for meaningful aggregation.

        Args:
            ia_value_dct (dict): Dictionary containing 'mean' and 'std' arrays for IA values over samples.
            base_value_dct (dict): Dictionary containing 'mean' and 'std' arrays for base values over samples.

        Returns:
            tuple (dict): Four dictionaries:
                          - IA absolute mean and std
                          - IA raw mean and std
                          - Base absolute mean and std
                          - Base raw mean and std
        """
        # Extract IA values
        ia_mean = ia_value_dct['mean']  # Shape: (samples, combinations)

        # IA absolute mean and std
        ia_abs_mean_across_samples = {
            'mean': np.mean(np.abs(ia_mean), axis=0),  # Mean of absolute values, shape: (combinations,)
            'std': np.std(np.abs(ia_mean), axis=0)  # Std of absolute values, shape: (combinations,)
        }

        # IA raw mean and std
        ia_raw_mean_across_samples = {
            'mean': np.mean(ia_mean, axis=0),  # Mean of raw values, shape: (combinations,)
            'std': np.std(ia_mean, axis=0)  # Std of raw values, shape: (combinations,)
        }
        print()

        # Extract base values
        base_mean = base_value_dct['mean']  # Shape: (samples,)

        # Base absolute mean and std
        base_abs_mean_across_samples = {
            'mean': np.mean(np.abs(base_mean)),  # Scalar
            'std': np.std(np.abs(base_mean))  # Scalar
        }

        # Base raw mean and std
        base_raw_mean_across_samples = {
            'mean': np.mean(base_mean),  # Scalar
            'std': np.std(base_mean)  # Scalar
        }

        return ia_abs_mean_across_samples, ia_raw_mean_across_samples, base_abs_mean_across_samples, base_raw_mean_across_samples

    def sample_aggregated_results(self, ia_value_dct, base_value_dct, num_samples):
        """
        Takes a random sample of the aggregated results across samples for each feature combination.

        Args:
            ia_value_dct (dict): Aggregated IA values dictionary with an extra layer of hierarchy (feature combinations).
                                 Structure: {combination: {'mean': ..., 'std': ...}, ...}.
            base_value_dct (dict): Aggregated base values dictionary with 'mean' and 'std' for each sample.
            num_samples (int): Number of samples to randomly select.

        Returns:
            dict: A dictionary with sampled aggregated mean and standard deviation for IA and base values.
        """
        # Determine the total number of samples available
        total_samples = next(iter(ia_value_dct.values()))['mean'].shape[0]  # Assuming all combinations have the same sample count

        # Select random indices for sampling
        sampled_indices = self.rng_.choice(total_samples, size=num_samples, replace=False)

        # Initialize dictionaries to store sampled IA and base values
        sampled_ia_values = {}
        sampled_base_values = {}

        # Sample from each feature combination in ia_value_dct
        for combination, stats in ia_value_dct.items():
            # Extract and sample mean and std values for the current combination
            ia_mean_sampled = stats['mean'][sampled_indices]
            ia_std_sampled = stats['std'][sampled_indices]

            # Store the sampled values in the result dictionary
            sampled_ia_values[combination] = {'mean': ia_mean_sampled, 'std': ia_std_sampled}

        # Sample base values based on selected indices
        sampled_base_values['mean'] = base_value_dct['mean'][sampled_indices]
        sampled_base_values['std'] = base_value_dct['std'][sampled_indices]

        return sampled_ia_values, sampled_base_values

    def map_combo_to_features(self, ia_mean_across_samples, feature_index_mapping: dict, combo_index_mapping: dict):
        """
        Maps the IA values (aggregated or non-aggregated across samples) to combinations of feature names.

        Args:
            ia_mean_across_samples (dict): Dictionary with keys 'mean' and 'std'. Each key can be a 1D array (shape: (num_combinations,))
                                           or a 2D array (shape: (num_samples, num_combinations)).

        Returns:
            dict: A dictionary where keys are tuples of feature names, and values are dicts. If the input is 1D,
                  each value is a dict with 'mean' and 'std' values. If the input is 2D, each value is a dict with
                  lists of sample values for 'mean' and 'std'.
        """
        result = {}
        mean_array = ia_mean_across_samples['mean']
        std_array = ia_mean_across_samples['std']

        # Match the structure based on the dimensions of the mean array
        match mean_array.ndim:
            case 1:  # 1D array: Aggregated across samples
                num_combinations = len(mean_array)
                for combination_index in range(num_combinations):
                    # Get the feature index combination from combo_index_mapping
                    feature_indices = combo_index_mapping[combination_index]

                    # Map feature indices to feature names using feature_index_mapping
                    feature_names = tuple(feature_index_mapping[idx] for idx in feature_indices)

                    # Get the mean and std IA values for this combination
                    mean_value = mean_array[combination_index]
                    std_value = std_array[combination_index]

                    # Store in the result dictionary
                    result[feature_names] = {'mean': mean_value, 'std': std_value}

            case 2:  # 2D array: Not aggregated, includes values for each sample
                num_samples, num_combinations = mean_array.shape
                for combination_index in range(num_combinations):
                    # Get the feature index combination from combo_index_mapping
                    feature_indices = combo_index_mapping[combination_index]

                    # Map feature indices to feature names using feature_index_mapping
                    feature_names = tuple(feature_index_mapping[idx] for idx in feature_indices)

                    # Get the mean and std IA values for this combination across all samples
                    mean_values = mean_array[:, combination_index]  # Array of means for each sample
                    std_values = std_array[:, combination_index]  # Array of stds for each sample

                    # Store in the result dictionary
                    result[feature_names] = {'mean': mean_values, 'std': std_values}

            case _:  # Handle unexpected array dimensions
                raise ValueError("Expected mean array to be 1D or 2D, but got an array with shape: "
                                 f"{mean_array.shape}")
        return result

    def get_top_interactions(self, abs_mapped_results_agg_samples, mapped_results_agg_samples, mapped_results_no_agg_samples,
                             top_n=20):
        """
        Extracts the top N most important interactions for each interaction order based on the absolute magnitude
        of the mean IA values, as well as the top N interactions without taking absolute values. Filters the
        second parameter based on these top interactions.

        Args:
            abs_mapped_results_agg_samples (dict): Aggregated absolute IA values across samples, mapping feature tuples
                                                   to dicts with 'mean' and 'std' values.
            mapped_results_agg_samples (dict): Aggregated raw IA values across samples, mapping feature tuples to dicts
                                               with 'mean' and 'std' values.
            mapped_results_no_agg_samples (dict): Non-aggregated IA values across samples, mapping feature tuples to
                                                  arrays of IA values per sample.
            top_n (int): Number of top interactions to extract for each interaction order.

        Returns:
            dict: Dictionary with keys 'top_abs_interactions', 'top_raw_interactions', 'top_abs_interactions_of_sample',
                  'top_raw_interactions_of_sample'.
        """
        # Step 1: Group the interactions by their order (length of the feature tuple) for absolute values
        abs_order_dict = defaultdict(list)
        for feature_tuple, value_dict in abs_mapped_results_agg_samples.items():
            order = len(feature_tuple)
            mean_value = value_dict['mean']
            std_value = value_dict['std']
            abs_order_dict[order].append((feature_tuple, mean_value, std_value))

        # Step 2: Group the interactions by their order for raw values
        raw_order_dict = defaultdict(list)
        for feature_tuple, value_dict in mapped_results_agg_samples.items():
            order = len(feature_tuple)
            mean_value = value_dict['mean']
            std_value = value_dict['std']
            raw_order_dict[order].append((feature_tuple, mean_value, std_value))

        # Step 3: Extract the top N interactions for absolute values
        abs_res = {}
        abs_selected_interactions = set()  # To track selected feature tuples for absolute values
        for order, interactions in abs_order_dict.items():
            # Sort interactions by absolute value of mean, in descending order
            interactions_sorted = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)

            # Select the top N interactions
            top_interactions = interactions_sorted[:top_n]
            abs_selected_interactions.update([feat_tuple for feat_tuple, _, _ in top_interactions])

            # Build a dict for this order, mapping from feature tuple to a dict of mean and std values
            abs_res[f'order_{order}'] = {
                feat_tuple: {'mean': mean_val, 'std': std_val} for feat_tuple, mean_val, std_val in top_interactions
            }

        # Step 4: Extract the top N interactions for raw values
        raw_res = {}
        raw_selected_interactions = set()  # To track selected feature tuples for raw values
        for order, interactions in raw_order_dict.items():
            # Sort interactions by absolute value of mean, in descending order
            interactions_sorted = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)

            # Select the top N interactions (preserves both positive and negative values)
            top_interactions = interactions_sorted[:top_n]

            raw_selected_interactions.update([feat_tuple for feat_tuple, _, _ in top_interactions])

            # Build a dict for this order, mapping from feature tuple to a dict of mean and std values
            raw_res[f'order_{order}'] = {
                feat_tuple: {'mean': mean_val, 'std': std_val} for feat_tuple, mean_val, std_val in top_interactions
            }

        # Step 5: Filter mapped_results_no_agg_samples for absolute and raw interactions
        abs_filtered_results = {
            feat_tuple: value_dict for feat_tuple, value_dict in mapped_results_no_agg_samples.items()
            if feat_tuple in abs_selected_interactions
        }

        raw_filtered_results = {
            feat_tuple: value_dict for feat_tuple, value_dict in mapped_results_no_agg_samples.items()
            if feat_tuple in raw_selected_interactions
        }

        # Step 6: Combine the results into the output dictionary
        final_res = {
            'top_abs_interactions': abs_res,  # Top absolute interactions from abs_mapped_results_agg_samples
            'top_raw_interactions': raw_res,  # Top raw interactions from mapped_results_agg_samples
            'top_abs_interactions_of_sample': abs_filtered_results,  # Filtered non-aggregated results for abs
            'top_raw_interactions_of_sample': raw_filtered_results  # Filtered non-aggregated results for raw
        }

        return final_res

    def get_most_interacting_features(self, abs_mapped_results_agg_samples):
        """
        Identifies the most interacting features across interaction orders greater than 1.

        For each feature, sums the absolute values of the mean IA values of all combinations
        of orders greater than 1 (i.e., interactions of order 2 and higher) in which the feature is involved.

        Args:
            mapped_results (dict): Output from map_combo_to_features, mapping tuples of feature names
                                   to dicts with 'mean' and 'std' values.

        Returns:
            dict: Dictionary mapping feature names to their total summed absolute IA values from higher-order interactions.
        """
        feature_interaction_sums = {}

        # Loop over all interactions
        for feature_tuple, value_dict in abs_mapped_results_agg_samples.items():
            # Only consider interactions of order greater than 1
            if len(feature_tuple) > 1:
                mean_value = value_dict['mean']
                abs_mean_value = abs(mean_value)
                # For each feature in the tuple, add the abs mean value to their total
                for feature in feature_tuple:
                    feature_interaction_sums[feature] = feature_interaction_sums.get(feature, 0.0) + abs_mean_value

        # Optionally, sort the features by their total summed value in descending order
        sorted_features = dict(sorted(feature_interaction_sums.items(), key=lambda item: item[1], reverse=True))

        return sorted_features

if __name__ == "__main__":
    print("Hello")
    summarizer = ClusterSummarizer(base_result_dir="../../results/local_tests/srmc/all/wb_state/", num_reps=3)
    summarizer.aggregate()





