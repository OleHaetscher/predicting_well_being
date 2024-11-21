import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import shapiq
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from shapiq import InteractionValues
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer


class RFRAnalyzer(BaseMLAnalyzer):
    """
    This class is the specific implementation of the random forest regression using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Inherits from
    BaseMLAnalyzer. For class attributes, see BaseMLAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, var_cfg, output_dir, df, rep, rank):
        """
        Constructor method of the RFRAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(var_cfg, output_dir, df, rep, rank)
        self.model = RandomForestRegressor(
            random_state=self.var_cfg["analysis"]["random_state"]
        )
        self.rng_ = np.random.RandomState(self.var_cfg["analysis"]["random_state"])  # Local RNG

    def calculate_shap_ia_values(self, X_scaled: pd.DataFrame, pipeline: Pipeline, combo_index_mapping: dict):
        """
        This method computes SHAP interaction values
        see Fumagalli et al. (2023) and Muschalik et al. (2024) for more details

        Args:
            X_scaled:
            pipeline:
            combo_index_mapping:

        Returns:
            dict:
            list(float):

        """
        # Parallelize the calculations processing chunks of the data
        n_jobs = self.var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"]
        chunk_size = X_scaled.shape[0] // n_jobs + (X_scaled.shape[0] % n_jobs > 0)
        print("n_jobs shap ia _values")
        print("chunk_size:", chunk_size)
        results = Parallel(n_jobs=n_jobs, verbose=1, backend=self.joblib_backend)(
            delayed(self.compute_shap_ia_values_for_chunk)(
                pipeline.named_steps["model"].regressor_,
                X_scaled[i: i + chunk_size]
            )
            for i in range(0, X_scaled.shape[0], chunk_size)
        )
        print("len results:", len(results))
        combined_results = [item for sublist in results for item in sublist]

        ia_values_arr, base_values_arr = self.process_shap_ia_values(
            results=combined_results,
            combo_index_mapping=combo_index_mapping,
            num_samples=X_scaled.shape[0],
        )

        return ia_values_arr, base_values_arr

    def process_shap_ia_values(self, results: list, combo_index_mapping: dict, num_samples: int):
        """
        In this function, we process the results obtained from the parallel execution of the shap ia values computations
        The "results" in "calculate_shap_ia_values" contain all necessary data to recreate the SHAP-IQ Interaction Value object
            - dict_values (dict with feature indice tuples as keys and shap-iq-values as values)
            - baseline_value
        The rest is given in the config or by e.g. the general dataframe
        we then create a mapping based on min_order and max_order that maps indices to feature combinations
        Returns:
            2darrray: np.array containing the shap ia values for each sample (axis 0) and each feature_combination (axis 1)
            1darray: np.array containing the the base values for each sample
        """
        base_values_arr = np.array([sample.baseline_value for sample in results])

        combo_index_mapping_reverse = {v: k for k, v in combo_index_mapping.items()}

        ia_values_dct = [sample.dict_values for sample in results]
        ia_values_arr = np.zeros((num_samples, len(combo_index_mapping)), dtype=np.float32)
        for sample_idx, sample_vals in enumerate(ia_values_dct):
            for feature_combo, value in sample_vals.items():
                idx = combo_index_mapping_reverse[feature_combo]
                ia_values_arr[sample_idx, idx] += value

        return ia_values_arr, base_values_arr

    def compute_shap_ia_values_for_chunk(self, model, X_subset) -> list:
        """
        This function computes (pairwise) SHAP interaction values for the rfr

        Args:
            explainer: shap.TreeExplainer
            X_subset: df, subset of X for which interaction values are computed, can be parallelized

        Returns:
            list: List of SHAP-IQ "InteractionValues" Objects
        """
        self.logger.log(f"Currently processing these indices: {X_subset.index[:3]}")
        self.logger.log(f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}")
        # SHAP IA Value computations
        explainer = shapiq.TreeExplainer(
            model=model,
            index=self.var_cfg["analysis"]["shap_ia_values"]["interaction_index"],
            min_order=self.var_cfg["analysis"]["shap_ia_values"]["min_order"],
            max_order=self.var_cfg["analysis"]["shap_ia_values"]["max_order"],
        )
        ia_value_lst = []
        for idx, x in X_subset.iterrows():
            ia_values_obj = explainer.explain(x.values)
            ia_value_lst.append(ia_values_obj)
        self.logger.log(f"SHAP IA computations for indices {X_subset.index[:3]} COMPLETED")

        return ia_value_lst

    def process_all_shap_ia_values(self):
        """
        Process SHAP interaction values by aggregating across repetitions and imputations.
        """
        # This method should only be executed after all repetitions have been completed
        # Ensure that it is not run per repetition when repetitions are run independently
        # So, we can check if MPI is used and rank == 0, or if MPI is not used and we're running a dedicated postprocessing step
        if not self.split_reps:
            if self.rank == 0 and self.var_cfg["analysis"]["shap_ia_values"][
                "comp_shap_ia_values"]:
                # Initialize dictionaries to collect results across repetitions
                ia_values_all_reps = {}
                base_values_all_reps = {}

                # Determine the number of repetitions
                num_reps = self.var_cfg["analysis"]["num_reps"]

                # Loop over all repetitions and load stored SHAP IA values
                for rep in range(num_reps):
                    shap_ia_values_filename_cluster = os.path.join(
                        self.spec_output_path,
                        f"{self.var_cfg['analysis']['output_filenames']['shap_ia_values_for_cluster']}_rep_{rep}.pkl"
                    )

                    if os.path.exists(shap_ia_values_filename_cluster):
                        with open(shap_ia_values_filename_cluster, "rb") as file:
                            shap_ia_results_reps_imps = pickle.load(file)

                        ia_values_all_reps[f"rep_{rep}"] = shap_ia_results_reps_imps["shap_ia_values"]
                        base_values_all_reps[f"rep_{rep}"] = shap_ia_results_reps_imps["base_values"]
                    else:
                        print(f"Warning: SHAP IA values file for repetition {rep} not found.")
                        continue

                # Now proceed with aggregation across repetitions and imputations
                ia_values_agg_reps_imps, base_value_agg_reps_imps = self.agg_ia_values_across_reps(
                    ia_value_dct=ia_values_all_reps,
                    base_value_dct=base_values_all_reps,
                )
                self.shap_ia_results_reps_imps["shap_ia_values"] = ia_values_agg_reps_imps
                self.shap_ia_results_reps_imps["base_values"] = base_value_agg_reps_imps

                # Aggregate results across samples
                abs_ia_values_agg_reps_imps_samples, base_value_agg_reps_imps_samples = self.agg_results_across_samples(
                    ia_value_dct=ia_values_agg_reps_imps.copy(),
                    base_value_dct=base_value_agg_reps_imps.copy(),
                )

                # Map combo to features
                abs_ia_values_agg_reps_imps_samples = self.map_combo_to_features(abs_ia_values_agg_reps_imps_samples)
                ia_values_agg_reps_imps = self.map_combo_to_features(ia_values_agg_reps_imps)

                # Sample aggregated results
                ia_values_sample, base_values_sample = self.sample_aggregated_results(
                    ia_values_agg_reps_imps,
                    base_value_agg_reps_imps,
                    num_samples=self.var_cfg["analysis"]["shap_ia_values"]["num_samples"]
                )

                self.shap_ia_results_processed["ia_values_sample"] = ia_values_sample
                self.shap_ia_results_processed["base_values_sample"] = base_values_sample

                # Get top interactions
                top_interactions_per_order = self.get_top_interactions(
                    abs_mapped_results_agg_samples=abs_ia_values_agg_reps_imps_samples,
                    mapped_results_no_agg_samples=ia_values_agg_reps_imps
                )
                # Get most interacting features
                top_interacting_features = self.get_most_interacting_features(
                    abs_mapped_results_agg_samples=abs_ia_values_agg_reps_imps_samples,
                )
                self.shap_ia_results_processed["top_interactions"] = top_interactions_per_order
                self.shap_ia_results_processed["top_interacting_features"] = top_interacting_features
                self.shap_ia_results_processed['abs_ia_value_agg_reps_imps_samples'] = abs_ia_values_agg_reps_imps_samples
                print("Processed all SHAP IA values.")

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
        Note: We must take the absolute mean of the ia_values, otherwise this aggregation makes no sense 

        Args:
            aggregated_results (dict): Output from the previous function, containing 'ia_values' and 'base_values',
                                       each with 'mean' and 'std' arrays over samples.

        Returns:
            tuple (dict) : Tuple of dicts with the ia_values and the base_values
        """  # TODO Does this makes sens for the base values?
        # Extract IA values
        ia_mean = ia_value_dct['mean']  # Shape: (samples, combinations)

        # Aggregate IA mean and std across samples (axis=0)
        ia_mean_across_samples = {
            'mean': np.mean(np.abs(ia_mean), axis=0),  # Mean of absolute values, shape: (combinations,)
            'std': np.std(np.abs(ia_mean), axis=0)  # Std of absolute values, shape: (combinations,)
        }

        # Extract base values
        base_mean = base_value_dct['mean']  # Shape: (samples,)

        # Aggregate base mean and std across samples
        base_mean_across_samples = {
            'mean': np.mean(base_mean),  # Scalar
            'std': np.std(base_mean)  # Scalar
        }

        return ia_mean_across_samples, base_mean_across_samples

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

    def map_combo_to_features(self, ia_mean_across_samples):
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
                    feature_indices = self.combo_index_mapping[combination_index]

                    # Map feature indices to feature names using feature_index_mapping
                    feature_names = tuple(self.feature_index_mapping[idx] for idx in feature_indices)

                    # Get the mean and std IA values for this combination
                    mean_value = mean_array[combination_index]
                    std_value = std_array[combination_index]

                    # Store in the result dictionary
                    result[feature_names] = {'mean': mean_value, 'std': std_value}

            case 2:  # 2D array: Not aggregated, includes values for each sample
                num_samples, num_combinations = mean_array.shape
                for combination_index in range(num_combinations):
                    # Get the feature index combination from combo_index_mapping
                    feature_indices = self.combo_index_mapping[combination_index]

                    # Map feature indices to feature names using feature_index_mapping
                    feature_names = tuple(self.feature_index_mapping[idx] for idx in feature_indices)

                    # Get the mean and std IA values for this combination across all samples
                    mean_values = mean_array[:, combination_index]  # Array of means for each sample
                    std_values = std_array[:, combination_index]  # Array of stds for each sample

                    # Store in the result dictionary
                    result[feature_names] = {'mean': mean_values, 'std': std_values}

            case _:  # Handle unexpected array dimensions
                raise ValueError("Expected mean array to be 1D or 2D, but got an array with shape: "
                                 f"{mean_array.shape}")

        return result

    def get_top_interactions(self, abs_mapped_results_agg_samples, mapped_results_no_agg_samples, top_n=20):
        """
        Extracts the top N most important interactions for each interaction order based on the absolute magnitude
        of the mean IA values, and then filters the second parameter based on these top interactions.

        Args:
            abs_mapped_results_agg_samples (dict): Aggregated absolute IA values across samples, mapping feature tuples
                                                   to dicts with 'mean' and 'std' values.
            mapped_results_no_agg_samples (dict): Non-aggregated IA values across samples, mapping feature tuples to
                                                  arrays of IA values per sample.
            top_n (int): Number of top interactions to extract for each interaction order.

        Returns:
            dict: Dictionary with keys 'order_1', 'order_2', ..., each containing a dict mapping feature name tuples
                  to mean IA values from abs_mapped_results_agg_samples and sample-specific IA values from
                  mapped_results_no_agg_samples.
        """
        # Step 1: Group the interactions by their order (length of the feature tuple) in abs_mapped_results_agg_samples
        order_dict = defaultdict(list)
        for feature_tuple, value_dict in abs_mapped_results_agg_samples.items():
            order = len(feature_tuple)
            mean_value = value_dict['mean']
            std_value = value_dict['std']
            order_dict[order].append((feature_tuple, mean_value, std_value))

        # Step 2: For each order, extract the top N interactions based on absolute mean IA values
        res = {}
        selected_interactions = set()  # To keep track of selected feature tuples across orders
        for order, interactions in order_dict.items():
            # Sort interactions by absolute value of mean, in descending order
            interactions_sorted = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)

            # Select the top N interactions and store them in selected_interactions set
            top_interactions = interactions_sorted[:top_n]
            selected_interactions.update([feat_tuple for feat_tuple, _, _ in top_interactions])

            # Build a dict for this order, mapping from feature tuple to a dict of mean and std values
            top_interactions_dict = {
                feat_tuple: {'mean': mean_val, 'std': std_val} for feat_tuple, mean_val, std_val in top_interactions
            }
            res[f'order_{order}'] = top_interactions_dict

        # Step 3: Filter mapped_results_no_agg_samples based on the selected interactions
        filtered_mapped_results = {
            feat_tuple: value_dict for feat_tuple, value_dict in mapped_results_no_agg_samples.items()
            if feat_tuple in selected_interactions
        }

        # Step 4: Combine the results into the output dictionary
        final_res = {
            'top_interactions': res,  # Top interactions from abs_mapped_results_agg_samples
            'filtered_samples': filtered_mapped_results  # Filtered non-aggregated results
        } # Note: Top interactions are the abs ia values

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
