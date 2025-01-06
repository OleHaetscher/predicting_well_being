import os
from typing import Union

import numpy as np
import pandas as pd
from src.utils.DataLoader import DataLoader
import openpyxl
import json
import pingouin as pg

from src.utils.utilfuncs import apply_name_mapping, format_df


class DescriptiveStatistics:
    """
    This class computes descriptives as specified in the PreReg. This includes
        - M and SD of all features used in the ML-based analysis
        - descriptive statistics for the individual wb_items and the wb_scores (criteria)
            - M and SD
            - ICC1s and ICC2s
            - Within-person and between-person correlations
        - Reliability of the wb_scores per dataset
    """
    def __init__(self, fix_cfg, var_cfg, name_mapping):
        self.var_cfg = var_cfg
        self.desc_cfg = self.var_cfg["postprocessing"]["descriptives"]
        self.data_loader = DataLoader()
        self.fix_cfg = fix_cfg
        self.name_mapping = name_mapping
        self.datasets = self.var_cfg["general"]["datasets_to_be_included"]

        self.esm_id_col_dct = self.var_cfg["preprocessing"]["esm_id_col"]
        self.esm_tp_col_dct = self.var_cfg["preprocessing"]["esm_timestamp_col"]

        self.full_data_path = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], "full_data")
        self.data_base_path = self.var_cfg['analysis']["path_to_preprocessed_data"]
        self.desc_results_base_path = self.var_cfg['postprocessing']["descriptives"]["base_path"]

    def create_m_sd_feature_table(self):
        """
        This method creates a table containing the mean (M) and standard deviation (SD) for continuous variables,
        and frequencies (counts and percentages) for binary variables. The table is subdivided based on the name
        of the features (visible by the prefix of the column names: 'pl_', 'sens_', 'srmc_', 'mac_').
        Note: Currently we do this for the total data, not per sample

        Returns:
            final_table: pd.DataFrame containing the formatted table in APA style.
        """
        full_df = self.data_loader.read_pkl(self.full_data_path)
        results = []
        feature_cats = self.var_cfg["analysis"]["feature_combinations"]

        for cat in feature_cats:
            prefix = f"{cat}_"
            prefixed_cols = [col for col in full_df.columns if col.startswith(prefix)]
            if not prefixed_cols:
                continue

            df_subset = full_df[prefixed_cols]
            binary_vars, continuous_vars = self.separate_binary_continuous_cols(df_subset)

            # Compute M / SD for continuous variables
            if continuous_vars:
                cont_stats = self.calculate_cont_descriptive_stats(
                    df=df_subset,
                    continuous_vars=continuous_vars,
                    prefix=prefix,
                    stats=self.desc_cfg["cont_agg_dct"],  # M / SD
                    var_as_index=False,
                )
                results.append(cont_stats)

            # Compute percentages for binary variables
            if binary_vars:
                bin_stats = self.calculate_bin_stats(
                    df=df_subset,
                    binary_vars=binary_vars,
                    prefix=prefix,
                    stats=self.desc_cfg["bin_agg_lst"],  # %
                )
                results.append(bin_stats)

        # Combine all results
        final_table = pd.concat(results, ignore_index=True)

        # Format dataframe
        final_table = format_df(
            df=final_table,
            capitalize=False,
            decimals=2
        )

        # Rename feature categories and reorder the columns
        final_table["Group"] = final_table["Group"].replace(self.name_mapping["category_names"])
        final_table = final_table[["Group"] + [col for col in final_table.columns if col != "Group"]]

        # Create a joint column containing M (SD) and %
        final_table['M (SD)'] = final_table.apply(lambda row: f"{row['M']} ({row['SD']})", axis=1)
        final_table['M (SD) / %'] = final_table.apply(
            lambda row: row['M (SD)'] if pd.notna(row['M']) else row['%'],
            axis=1
        )
        final_table = final_table.drop(columns=["M", "SD", "%", "M (SD)"])

        final_table["scale_endpoints"] = final_table["Variable"].apply(
            lambda feat: self.get_scale_endpoints(self.fix_cfg.copy(), feat)
        )

        # Rename Features
        final_table["Variable"] = apply_name_mapping(
            features=list(final_table["Variable"]),
            name_mapping=self.name_mapping,
            prefix=True
        )

        final_table = final_table.reset_index(drop=True)
        print()

        # If defined in the cfg, store results
        if self.desc_cfg["store"]:
            self.save_file(
                data=final_table,
                file_name="predictors_m_sd",
                file_path=self.desc_results_base_path,
                filetype="xlsx"
            )

    def get_scale_endpoints(self, data: dict, feature_name: str) -> Union[np.nan, tuple[float, float]]:
        """
        Recursively searches a nested dictionary/list structure to find an entry
        where "name" == target_name and returns the "scale_endpoints" as a (min, max) tuple.

        Args:
            data:
            feature_name:

        Returns:
            np.nan if no match is found, or the scale endpoints as a tuple (min, max).

        """
        # Remove prefix
        if any(feature_name.startswith(prefix) for prefix in ["pl_", "srmc_", "mac_", "sens:"]):
            feature_name_no_prefix = feature_name.split('_', 1)[-1]
        else:
            feature_name_no_prefix = feature_name

        if isinstance(data, dict):
            # Check if this dictionary node has the target feature name
            if data.get("name") == feature_name_no_prefix:
                scale_endpoints = data.get("scale_endpoints")
                if scale_endpoints is not None:
                    return (scale_endpoints["min"], scale_endpoints["max"])

            # If not matched at this level, recursively search all values
            for key, value in data.items():
                result = self.get_scale_endpoints(value, feature_name_no_prefix)
                if result is not None:
                    return result

        elif isinstance(data, list):
            # If data is a list, search each element
            for item in data:
                result = self.get_scale_endpoints(item, feature_name_no_prefix)
                if result is not None:
                    return result

        # If we reach here, no match was found in this branch
        return None
            
    @staticmethod
    def calculate_cont_descriptive_stats(df, continuous_vars, stats, var_as_index=True, prefix=None):
        """
        Calculate continuous statistics for specified variables with custom aggregation mapping.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            continuous_vars (list): List of continuous variable names.
            prefix (str): Prefix to remove from variable names for display.
            stats (dict): Aggregation functions as keys and their desired output names as values.

        Returns:
            pd.DataFrame: A DataFrame with calculated statistics in APA format.
        """
        # Extract the list of aggregation functions from the stats dictionary keys
        agg_funcs = list(stats.keys())

        # Apply the aggregation functions to the continuous variables
        cont_stats = df[continuous_vars].agg(agg_funcs).transpose()

        if not var_as_index:
            cont_stats = cont_stats.reset_index()
            cont_stats.columns = ['Variable'] + [stats[func] for func in agg_funcs]
        else:
            cont_stats.columns = [stats[func] for func in agg_funcs]

        if prefix:
            cont_stats['Group'] = prefix.rstrip('_')

        return cont_stats

    @staticmethod
    def calculate_bin_stats(df, binary_vars, prefix, stats):
        """
        Calculate binary statistics for specified variables with custom aggregation mapping.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            binary_vars (list): List of binary variable names to analyze.
            prefix (str): Prefix to remove from variable names for display.
            stats (list): Mapping of custom statistics to calculate, with keys as functions
                          (e.g., 'value_counts') and values as column names.

        Returns:
            pd.DataFrame: A DataFrame with binary statistics, including frequency, total count,
                          and percentage of occurrences, formatted in APA style.
        """
        bin_stats = df[binary_vars].apply(lambda x: x.value_counts(dropna=True)).transpose()
        # bin_stats['Variable'] = bin_stats.index.str.replace(prefix, '', regex=False)
        bin_stats['Variable'] = bin_stats.index  # .str.replace(prefix, '', regex=False)

        if "%" in stats:
            bin_stats['Frequency'] = np.nan

            # Handle counts for '1' (could be 1 or 1.0)
            if 1 in bin_stats.columns:
                bin_stats['Frequency'] = bin_stats[1]
            elif 1.0 in bin_stats.columns:
                bin_stats['Frequency'] = bin_stats[1.0]

            # Calculate percentage of '1' occurrences, handling NaN frequencies properly
            bin_stats['Total'] = df[binary_vars].notna().sum().values
            bin_stats['%'] = (bin_stats['Frequency'] / bin_stats['Total']) * 100
            bin_stats['Group'] = prefix.rstrip('_')
            bin_stats = bin_stats[['Group', 'Variable', '%']].reset_index(drop=True)

        return bin_stats

    @staticmethod  # TODO Make class independent function
    def separate_binary_continuous_cols(df):
        """
        This function determines which features of the current data are continuous (i.e., containing not only 0/1 per column)
        and which features are binary (i.e., only containing 0/1 per column). This may also includes NaNs.
        It sets the resulting column lists as class attributes

        Returns:

        """
        binary_cols = list(df.columns[(df.isin([0, 1]) | df.isna()).all(axis=0)])
        continuous_cols = [col for col in df.columns if col not in binary_cols]  # ordered as in df
        return binary_cols, continuous_cols

    def create_wb_items_stats_per_dataset(self, dataset: str):
        """

        Args:
            dataset

        Returns:

        """
        wb_items_df = None
        result_dct = {}
        esm_id_col = self.var_cfg["preprocessing"]["esm_id_col"][dataset]
        esm_tp_col = self.var_cfg["preprocessing"]["esm_timestamp_col"][dataset]

        print("XX")
        path_to_state_df = os.path.join(self.data_base_path, f"wb_items_{dataset}")
        path_to_trait_df = os.path.join(self.data_base_path, f"trait_wb_items_{dataset}")

        try:
            state_df = self.data_loader.read_pkl(path_to_state_df)
            state_df = state_df.drop(columns=["wb_state", "pa_state", "na_state"], errors="ignore")

            if dataset == "emotions":
                state_df = self.merge_wb_items(state_df, prefix_a="occup_", prefix_b="int_")

            # Apply name mapping to state_df
            state_df.columns = apply_name_mapping(
                features=state_df.columns,
                name_mapping=self.name_mapping,
                prefix=False
            )
            # Reorder the columns
            state_wb_items_col_order = self.desc_cfg["wb_items"]["state_order"]
            wb_item_cols_ordered = [col for col in state_wb_items_col_order if col in state_df.columns]
            other_cols = [col for col in state_df if col not in state_wb_items_col_order]
            state_df = state_df[other_cols + wb_item_cols_ordered]

            state_df.columns = [f"state_{col}" if col not in [esm_id_col, esm_tp_col] else col
                                for col in state_df.columns]
        except FileNotFoundError:
            print(f"No state_df found for {dataset}")
            state_df = None

        try:
            trait_df = self.data_loader.read_pkl(path_to_trait_df)
            trait_df = trait_df.drop(columns=["wb_trait", "pa_trait", "na_trait"], errors="ignore")
            # Apply name mapping to state_df
            trait_df.columns = apply_name_mapping(
                features=trait_df.columns,
                name_mapping=self.name_mapping,
                prefix=False
            )
            # Reorder the columns
            trait_wb_items_col_order = self.desc_cfg["wb_items"]["trait_order"]
            wb_item_cols_ordered = [col for col in trait_wb_items_col_order if col in trait_df.columns]
            other_cols = [col for col in trait_df if col not in trait_wb_items_col_order]
            trait_df = trait_df[other_cols + wb_item_cols_ordered]

            trait_df.columns = [f"trait_{col}" for col in trait_df.columns]
        except FileNotFoundError:
            print(f"No trait_df found for {dataset}")
            trait_df = None

        # Concatenate state_df and trait_df if they exist
        if state_df is not None or trait_df is not None:
            wb_items_df = pd.concat([state_df, trait_df], axis=0, ignore_index=True)

            # Calculate descriptive statistics
            m_sd_df_wb_items = self.calculate_cont_descriptive_stats(
                df=wb_items_df.drop(columns=esm_id_col, errors="ignore"),
                continuous_vars=wb_items_df
                    .drop(columns=esm_id_col, errors="ignore")
                    .select_dtypes(include=[np.number])
                    .columns,
                stats=self.desc_cfg["cont_agg_dct"],
                var_as_index=True,
            )
            result_dct["m_sd"] = m_sd_df_wb_items

            # Calculate BP/WP statistics if state_df specific columns are available
            if state_df is not None and dataset in self.esm_id_col_dct and dataset in self.esm_tp_col_dct:
                wp_corr, bp_corr, icc1, icc2 = self.calc_bp_wp_statistics(
                    df=state_df,
                    id_col=self.esm_id_col_dct[dataset],
                    tp_col=self.esm_tp_col_dct[dataset],
                )
                result_dct["wp_corr"] = wp_corr
                result_dct["bp_corr"] = bp_corr
                result_dct["icc1"] = icc1
                result_dct["icc2"] = icc2

            if trait_df is not None:
                result_dct["trait_corr"] = trait_df.corr()
            else:
                result_dct["trait_corr"] = None

        return result_dct

    def merge_wb_items(self, state_df, prefix_a, prefix_b):
        """
        Merges columns in `state_df` that differ only by the specified prefixes.
        The new column will have the same name without the prefix and contain the mean of the two columns.

        Args:
            state_df (pd.DataFrame): The input DataFrame containing columns to merge.
            prefix_a (str): The first prefix to look for in column names.
            prefix_b (str): The second prefix to look for in column names.

        Returns:
            pd.DataFrame: A new DataFrame with merged columns.
        """
        # Ensure prefixes are strings
        if not isinstance(prefix_a, str) or not isinstance(prefix_b, str):
            raise ValueError("prefix_a and prefix_b must be strings.")

        # Extract column names that start with prefix_a and prefix_b
        cols_a = [col for col in state_df.columns if col.startswith(prefix_a)]
        cols_b = [col for col in state_df.columns if col.startswith(prefix_b)]

        # Remove prefixes to get the suffixes
        suffixes_a = {col[len(prefix_a):] for col in cols_a}
        suffixes_b = {col[len(prefix_b):] for col in cols_b}

        # Find common suffixes present in both prefix_a and prefix_b
        common_suffixes = suffixes_a.intersection(suffixes_b)

        if not common_suffixes:
            print("No common suffixes found between the provided prefixes.")
            return state_df.copy()

        # Initialize a copy of the DataFrame to avoid modifying the original
        merged_df = state_df.copy()

        for suffix in common_suffixes:
            col_a = prefix_a + suffix
            col_b = prefix_b + suffix
            new_col = suffix  # New column without prefix

            # Compute the mean of the two columns
            merged_df[new_col] = state_df[[col_a, col_b]].mean(axis=1)
            merged_df.drop([col_a, col_b], axis=1, inplace=True)

        return merged_df

    @staticmethod
    def calc_bp_wp_statistics(df, id_col, tp_col):
        """
        Calculates within-person (within-group) and between-person (between-group) correlations and ICCs for given variables.
        Based on the R implementation in psych (Revelle)

        Args:
            df: pd.DataFrame, data containing the variables and the grouping variable.
            id_col: str, the name of the column containing the group identifier.

        Returns:
            corr_W: pd.DataFrame, containing the pooled within-person correlations.
            corr_B: pd.DataFrame, containing the between-person correlations.
            ICC1_dict: dict, containing the ICC1 values for each variable.
            ICC2_dict: dict, containing the ICC2 values for each variable.
        """
        # Exclude the group identifier column to get the variables to correlate
        variables_to_correlate = df.columns.drop([id_col, tp_col], errors="ignore")

        # Total number of observations and number of variables
        N = len(df)
        p = len(variables_to_correlate)

        # Get unique groups
        groups = df[id_col].unique()
        K = len(groups)

        # Initialize dictionaries to store sums
        SSB_dict = {var: 0 for var in variables_to_correlate}
        SSW_dict = {var: 0 for var in variables_to_correlate}
        n_list = []

        # Overall mean per variable
        overall_mean = df[variables_to_correlate].mean()

        # Loop over each group to compute SSB and SSW per variable
        for group in groups:
            group_df = df[df[id_col] == group]
            n_i = len(group_df)
            n_list.append(n_i)
            group_means = group_df[variables_to_correlate].mean()

            # Between-group sum of squares
            for var in variables_to_correlate:
                SSB_dict[var] += n_i * (group_means[var] - overall_mean[var]) ** 2

            # Within-group sum of squares
            for var in variables_to_correlate:
                deviations = group_df[var] - group_means[var]
                SSW_dict[var] += (deviations ** 2).sum()

        # Degrees of freedom
        df_between = K - 1
        df_within = N - K
        n_bar = np.mean(n_list)  # Average group size

        # Initialize dictionaries to store mean squares and ICCs
        MSb_dict = {}
        MSw_dict = {}
        # Initialize empty Series for ICC1 and ICC2
        ICC1_series = pd.Series(dtype=float)
        ICC2_series = pd.Series(dtype=float)

        # Compute MSb and MSw per variable and then ICC1 and ICC2
        for var in variables_to_correlate:
            MSb = SSB_dict[var] / df_between
            MSw = SSW_dict[var] / df_within
            MSb_dict[var] = MSb
            MSw_dict[var] = MSw

            # Calculate ICC1 using the provided formula
            numerator = MSb - MSw
            denominator = MSb + MSw * (n_bar - 1)
            ICC1 = numerator / denominator if denominator != 0 else np.nan

            # ICC2 is the reliability of the group means
            ICC2 = (MSb - MSw) / MSb if MSb != 0 else np.nan

            # Store the results
            #ICC1_dict[var] = np.round(ICC1, 2)
            #ICC2_dict[var] = np.round(ICC2, 2)

            # Store the results in Series
            ICC1_series[var] = np.round(ICC1, 2)
            ICC2_series[var] = np.round(ICC2, 2)

        # Compute within-group and between-group correlation matrices as before
        # Initialize sums of squares and cross-products matrices
        variables_list = list(variables_to_correlate)
        p = len(variables_list)
        SSW = np.zeros((p, p))  # Within-group sum of squares and cross-products
        SSB = np.zeros((p, p))  # Between-group sum of squares and cross-products

        # Mean center the overall data (used for total covariance)
        overall_mean_vector = df[variables_to_correlate].mean().values

        N_total = 0

        for group in groups:
            group_df = df[df[id_col] == group]
            n_i = len(group_df)

            # Mean of the group
            mu_i = group_df[variables_to_correlate].mean().values

            # Deviations from the group mean
            group_centered = group_df[variables_to_correlate] - mu_i

            # Compute within-group sum of squares and cross-products matrix
            cov_within = group_centered.cov(ddof=0)
            SSW_i = cov_within.values * (n_i - 1)
            SSW += SSW_i

            # Compute between-group sum of squares and cross-products matrix
            delta_mu = (mu_i - overall_mean_vector).reshape(-1, 1)  # Column vector
            SSB_i = n_i * np.dot(delta_mu, delta_mu.T)
            SSB += SSB_i

            N_total += n_i  # Update total observations

        # Degrees of freedom
        df_within = N_total - K
        df_between = K - 1

        # Compute the pooled within-group covariance matrix
        S_W = SSW / df_within

        # Compute the between-group covariance matrix
        S_B = SSB / df_between

        # Compute within-group correlation matrix
        std_W = np.sqrt(np.diag(S_W))
        corr_W = S_W / np.outer(std_W, std_W)
        corr_W = pd.DataFrame(corr_W, index=variables_to_correlate, columns=variables_to_correlate)

        # Compute between-group correlation matrix
        std_B = np.sqrt(np.diag(S_B))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_B = np.where(np.outer(std_B, std_B) == 0, 0, S_B / np.outer(std_B, std_B))
        corr_B = pd.DataFrame(corr_B, index=variables_to_correlate, columns=variables_to_correlate)

        return corr_W, corr_B, ICC1_series, ICC2_series

    def compute_rel(self, dataset: str):
        """
        This function computes the reliability of the well-being measures. We return a pd.Series
        to facilitate creating a corr table

        Args:
            dataset (str): The dataset to compute the reliability

        Returns:
            pd.Series: Contains the criteria as indices and the reliabilities as values
        """
        # rel_dct = {}
        state_rel_series = pd.Series(dtype=float)
        trait_rel_series = pd.Series(dtype=float)
        crit_cols = self.fix_cfg["var_assignments"]["crit"]
        crit_avail_dct = self.var_cfg["analysis"]["crit_available"]

        for crit in crit_cols:
            if dataset not in crit_avail_dct[crit]:
                continue

            if "trait" in crit:
                trait_df = self.prepare_rel_data(dataset=dataset, crit_type="trait")
                rel = self.compute_internal_consistency(df=trait_df, construct=crit, dataset=dataset)
                trait_rel_series[crit] = rel
            elif "state" in crit:
                state_df = self.prepare_rel_data(dataset=dataset, crit_type="state")
                rel = self.compute_split_half_rel(df=state_df, construct=crit, dataset=dataset)
                state_rel_series[crit] = rel
                print(f"Rel {crit}: {rel}")
            else:
                raise ValueError("Unknown criterion")

        # Map state_crit_names
        trait_rel_series.index = apply_name_mapping(
            features=list(trait_rel_series.index),
            name_mapping=self.name_mapping,
            prefix=False
        )
        state_rel_series.index = apply_name_mapping(
            features=list(state_rel_series.index),
            name_mapping=self.name_mapping,
            prefix=False
        )

        return state_rel_series, trait_rel_series

    def prepare_rel_data(self, dataset: str, crit_type: str):
        """
        This function prepares the data to compute the reliability for a given dataset

        Args:
            dataset (str): given dataset
            crit_type (str): "trait" or "state"

        Returns:
            tuple(pd.DataFrame, dict): trait and state df processed for reliability analysis
        """
        df = None
        # state data
        if crit_type == "state":
            timestamp_col = self.var_cfg["preprocessing"]["esm_timestamp_col"][dataset]
            id_col = self.var_cfg["preprocessing"]["esm_id_col"][dataset]

            path_to_state_data = os.path.join(self.data_base_path, f"wb_items_{dataset}")
            df = self.data_loader.read_pkl(path_to_state_data)

            # Ensure timestamp is in datetime format
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

            # Create idx_measurement_per_person
            df['idx_measurement_per_person'] = df.sort_values(by=timestamp_col).groupby(id_col).cumcount()
            df["joint_user_id"] = df[id_col].apply(lambda x: f"{dataset}_{x}")
            df = df.sort_values([id_col, "idx_measurement_per_person"])
            columns_to_select = ["pa_state", "na_state", "wb_state", "idx_measurement_per_person", "joint_user_id"]
            df = df[df.columns.intersection(columns_to_select)]

        # trait data
        elif crit_type == "trait":
            path_to_trait_data = os.path.join(self.data_base_path, f"trait_wb_items_{dataset}")
            df = self.data_loader.read_pkl(path_to_trait_data)

        else:
            raise ValueError("crit type must be trait or state")

        return df

    def compute_internal_consistency(self, df, construct, dataset):
        """
        This function computes the internal consistency as a reliability measure of the trait wb_items

        Returns:

        """
        # Retrieve item configurations
        item_cfg = self.fix_cfg["person_level"]["criterion"]

        # Here the default value makes sense
        pa_trait_items = item_cfg[0]["item_names"].get(dataset, [])
        na_trait_items = item_cfg[1]["item_names"].get(dataset, [])

        # Select items based on the construct
        if construct == "pa_trait":
            selected_items = pa_trait_items
        elif construct == "na_trait":
            selected_items = na_trait_items
        elif construct == "wb_trait":
            df[na_trait_items] = self.inverse_code(
                df=df[na_trait_items],
                min_scale=1,
                max_scale=6
            )
            selected_items = pa_trait_items + na_trait_items
        else:
            raise ValueError("construct must be one of 'pa_trait', 'na_trait', or 'wb_trait'")

        # Filter the DataFrame
        df_filtered = df[selected_items] if selected_items else None

        # Compute Cronbach's alpha if valid data exists
        if df_filtered is not None and not df_filtered.empty and len(df_filtered.columns) > 1:
            alpha = pg.cronbach_alpha(data=df_filtered)[0]
        else:
            alpha = None

        return alpha

    @staticmethod
    def inverse_code(df: pd.DataFrame, min_scale: int, max_scale: int) -> pd.DataFrame:
        """
        Inverse codes the negative affect items by subtracting each value from the maximum value of the scale.

        Args:
            df (pd.DataFrame): The DataFrame containing the negative affect items.
            min_scale (int): The minimum value of the scale.
            max_scale (int): The maximum value of the scale.

        Returns:
            pd.DataFrame: DataFrame with inverse-coded negative affect items.
        """
        return max_scale + min_scale - df

    def compute_split_half_rel(self,
                               df: pd.DataFrame,
                               dataset: str,
                               construct: str,
                               user_id_col: str = "joint_user_id",
                               measurement_idx_col: str = "idx_measurement_per_person",
                               method: str = "individual_halves",
                               correlation_method: str = "spearman") -> float:
        """
        Computes the split-half reliability by splitting the data according to the specified method,
        then calculating the correlation of these halves across users.

        Args:
            df_dct (dict of pd.DataFrame): Dictionary of DataFrames containing the data.
            construct (str): The name of the column representing the construct.
            user_id_col (str): The column representing unique user IDs, default is "joint_user_id".
            measurement_idx_col (str): The column representing measurement indices for each person.
            method (str): The method for splitting the data, could be "odd_even" or "individual_halves".
            correlation_method (str): The method for calculating correlation ("pearson" or "spearman"). Default is "spearman".

        Returns:
            float: Weighted mean reliability across datasets.
        """
        # Check if data is present
        if construct not in df.columns:  # This may happen
            print(f"The construct '{construct}' is not a column in the DataFrame for dataset {dataset}.")
        if user_id_col not in df.columns:
            raise ValueError(f"The column '{user_id_col}' is not in the DataFrame for dataset {dataset}.")
        if measurement_idx_col not in df.columns:
            raise ValueError(f"The DataFrame for dataset {dataset} must have the column '{measurement_idx_col}'.")

        # Initialize lists to store the summed construct values for the two halves
        first_half_means = []
        second_half_means = []

        # Group by user_id_col and compute the sum of construct values for each half
        for user_id, group in df.groupby(user_id_col):
            group = group.sort_values(by=measurement_idx_col)  # Ensure measurements are ordered correctly

            if method == "odd_even":
                # Sum the construct values for odd and even indices
                odd_mean = group[group[measurement_idx_col] % 2 == 1][construct].mean()
                even_mean = group[group[measurement_idx_col] % 2 == 0][construct].mean()
                first_half_means.append(odd_mean)
                second_half_means.append(even_mean)
            elif method == "individual_halves":
                # Split the measurements into two halves
                n = len(group)
                half = n // 2
                first_half = group.iloc[:half]
                second_half = group.iloc[half:]

                first_half_mean = first_half[construct].mean()
                second_half_mean = second_half[construct].mean()

                first_half_means.append(first_half_mean)
                second_half_means.append(second_half_mean)
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'odd_even' or 'individual_halves'.")

        # Convert to pandas Series to calculate correlation
        first_half_series = pd.Series(first_half_means)
        second_half_series = pd.Series(second_half_means)

        # Drop NaN pairs (cases where a user might have NaN in one of the halves)
        valid_idx = (~first_half_series.isna()) & (~second_half_series.isna())
        first_half_series = first_half_series[valid_idx]
        second_half_series = second_half_series[valid_idx]

        # Check if we have enough data to compute correlation
        if len(first_half_series) < 2:
            print(f"Not enough data to compute correlation for dataset {dataset}.")
            reliability = np.nan
        else:
            correlation = first_half_series.corr(second_half_series, method=correlation_method)

            # Calculate the split-half reliability using the Spearman-Brown formula
            reliability = 2 * correlation / (1 + correlation) if correlation is not None else np.nan

        print(f"Reliability for dataset {dataset}: {np.round(reliability, 3)}")
        return reliability

    def create_wb_items_table(self,
                              dataset: str,
                              m_sd_df: pd.DataFrame,
                              icc1: pd.Series = None,
                              icc2: pd.Series = None,
                              rel: pd.Series = None,
                              bp_corr: pd.DataFrame = None,
                              wp_corr: pd.DataFrame = None,
                              trait_corr: pd.DataFrame = None):
        """
        Creates a classical item descriptives table in APA style. This includes:
            - A column with the item names (index from m_sd_df)
            - A column with M (SD)
            - A column with ICC1 (if provided)
            - A column with ICC2 (if provided)
            - A column with reliability (rel) if provided
            - A correlation table on the rightmost side:
                * For state items: Upper triangular from bp_corr and lower triangular from wp_corr
                * For trait items: Upper triangular from trait_corr

        The final table is indexed by the wb_items. All Series and DataFrames must share the same index.
        We only retain items present in m_sd_df to avoid extraneous empty columns.
        """

        # Use only the items present in m_sd_df as the final unified index
        unified_index = m_sd_df.index

        # Reindex main df if needed (should already match but just to be safe)
        table = m_sd_df.reindex(unified_index)

        # Join reliability, ICC1, ICC2 if provided
        if rel is not None:
            rel = rel.reindex(unified_index)
            table = table.join(rel.rename("Rel"), how="left")

        if icc1 is not None:
            icc1 = icc1.reindex(unified_index)
            table = table.join(icc1.rename("ICC1"), how="left")

        if icc2 is not None:
            icc2 = icc2.reindex(unified_index)
            table = table.join(icc2.rename("ICC2"), how="left")

        # Handle correlation matrices for state items (bp_corr and wp_corr)
        if bp_corr is not None and wp_corr is not None:
            # Reindex both correlation matrices
            bp_corr = bp_corr.reindex(index=unified_index, columns=unified_index)
            wp_corr = wp_corr.reindex(index=unified_index, columns=unified_index)

            # Create masks and combine
            n = len(unified_index)
            lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

            combined_values = bp_corr.values.copy()
            wp_values = wp_corr.values

            # Replace lower triangle with wp_corr values
            combined_values[lower_mask] = wp_values[lower_mask]

            combined_corr = pd.DataFrame(combined_values,
                                         index=unified_index,
                                         columns=unified_index)
            table = table.join(combined_corr, how="left", rsuffix="_corr")

        elif bp_corr is not None:
            bp_corr = bp_corr.reindex(index=unified_index, columns=unified_index)
            table = table.join(bp_corr, how="left", rsuffix="_bp")

        elif wp_corr is not None:
            wp_corr = wp_corr.reindex(index=unified_index, columns=unified_index)
            table = table.join(wp_corr, how="left", rsuffix="_wp")

        # Handle trait correlations (upper triangular only)
        if trait_corr is not None:
            trait_corr = trait_corr.reindex(index=unified_index, columns=unified_index)
            mask_lower = np.tril(np.ones((len(unified_index), len(unified_index)), dtype=bool), k=-1)
            trait_corr_values = trait_corr.values.copy()
            trait_corr_values[mask_lower] = np.nan
            trait_corr = pd.DataFrame(trait_corr_values,
                                      index=unified_index,
                                      columns=unified_index)
            table = table.join(trait_corr, how="left", rsuffix="_trait_corr")

        table = table.dropna(axis=1, how='all')

        # Format the table
        table = format_df(
            df=table,
            capitalize=False,
            decimals=2
        )

        if self.desc_cfg["store"]:
            self.save_file(
                data=table,
                filetype="xlsx",
                file_path=os.path.join(self.desc_results_base_path, dataset),
                file_name=f"wb_item_desc_table_{dataset}",
                index=True,
            )

    def save_file(self, data, filetype, file_path, file_name, index=None):
        """
        Save data to a specified file path and name in the given filetype.

        Parameters:
            data (DataFrame or dict): The data to save.
            filetype (str): The type of file to save ('xlsx' or 'json').
            file_path (str): The directory where the file will be saved.
            file_name (str): The name of the file to save (with extension).
        """
        # Ensure the directory exists
        os.makedirs(file_path, exist_ok=True)

        # Full file path
        full_file_path = os.path.join(file_path, f"{file_name}.{filetype}")

        # Save data based on filetype
        if filetype == "xlsx":
            if hasattr(data, "to_excel"):  # Check if data is a DataFrame
                data.to_excel(full_file_path, index=index)
            else:
                raise ValueError("Data must be a DataFrame for xlsx filetype.")
        elif filetype == "json":
            if isinstance(data, dict):  # Check if data is a dictionary
                with open(full_file_path, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                raise ValueError("Data must be a dictionary for json filetype.")
        else:
            raise ValueError(f"Unsupported filetype: {filetype}")














