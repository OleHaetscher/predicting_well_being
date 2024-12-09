import os
import numpy as np
import pandas as pd
from src.utils.DataLoader import DataLoader
import openpyxl
import json
import pingouin as pg


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

        Returns:
            final_table: pd.DataFrame containing the formatted table in APA style.
        """
        # TODO: Per Sample or in total? Currently only total, but we could easily modify this

        full_df = self.data_loader.read_pkl(self.full_data_path)
        results = []
        feature_cats = self.var_cfg["analysis"]["feature_combinations"]
        cont_agg_dct = {}
        bin_agg_lst = []

        for cat in feature_cats:
            prefix = f"{cat}_"
            prefixed_cols = [col for col in full_df.columns if col.startswith(prefix)]
            if not prefixed_cols:
                continue
            df_subset = full_df[prefixed_cols]

            binary_vars, continuous_vars = self.separate_binary_continuous_cols(df_subset)

            # Compute statistics for continuous variables
            if continuous_vars:
                cont_agg_dct = self.desc_cfg["cont_agg_dct"]
                cont_stats = self.calculate_cont_stats(
                    df=df_subset,
                    continuous_vars=continuous_vars,
                    prefix=prefix,
                    stats=cont_agg_dct
                )
                results.append(cont_stats)

            # Compute frequencies for binary variables
            if binary_vars:
                bin_agg_lst = self.desc_cfg["bin_agg_lst"]
                bin_stats = self.calculate_bin_stats(
                    df=df_subset,
                    binary_vars=binary_vars,
                    prefix=prefix,
                    stats=bin_agg_lst,
                )
                results.append(bin_stats)

        # Combine all results
        final_table = pd.concat(results, ignore_index=True)
        col_names = list(cont_agg_dct.keys()) + bin_agg_lst

        # Format numerical columns to two decimal places
        final_table = DescriptiveStatistics.format_columns(
            df=final_table,
            columns=col_names,  # ["M", "SD", "%"],
            decimals=2
        )

        # Rename feature names
        final_table["Group"] = final_table["Group"].replace(self.name_mapping["category_names"])
        for cat in feature_cats:
            final_table['Variable'] = final_table['Variable'].replace(self.name_mapping[cat])
        final_table = final_table.reset_index(drop=True)

        # If defined in the cfg, store results
        if self.desc_cfg["store"]:
            file_name = os.path.join(self.desc_results_base_path, "descriptives_table.xlsx")
            final_table.to_excel(file_name, index=False)

    @staticmethod
    def format_columns(df: pd.DataFrame, columns: list = None, decimals: int = 2) -> pd.DataFrame:
        """
        Formats specified numerical columns of a DataFrame to the given number of decimal places.
        If no columns are specified, all numerical columns are formatted.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            columns (list, optional): List of column names to format. Defaults to None.
            decimals (int): Number of decimal places to round to. Defaults to 2.

        Returns:
            pd.DataFrame: A DataFrame with the specified (or all numerical) columns rounded to the given number of decimal places.
        """
        if columns is None:
            # Select only numerical columns if no specific columns are provided
            columns = df.select_dtypes(include=['number']).columns.tolist()
        for column in columns:
            if column in df:
                df[column] = df[column].round(decimals)
        return df

    @staticmethod
    def calculate_cont_stats(df, continuous_vars, prefix, stats):
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
        cont_stats = df[continuous_vars].agg(agg_funcs).transpose().reset_index()

        # Rename the columns using the stats dictionary values
        # The columns after 'index' correspond to the aggregation functions
        cont_stats.columns = ['Variable'] + [stats[func] for func in agg_funcs]

        # Remove the prefix from the variable names
        cont_stats['Variable'] = cont_stats['Variable'].str.lstrip(prefix)

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
        bin_stats['Variable'] = bin_stats.index.str.replace(prefix, '', regex=False)

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
            bin_stats = bin_stats[['Group', 'Variable', 'Frequency', 'Total', '%']].reset_index(drop=True)

        return bin_stats

    @staticmethod
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

    def create_wb_items_statistics(self):
        """

        Returns:

        """
        full_df = self.data_loader.read_pkl(self.full_data_path)
        for dataset in self.var_cfg["general"]["datasets_to_be_included"]:
            print("XX")

            path_to_state_data = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], f"wb_items_{dataset}")
            state_df = self.data_loader.read_pkl(path_to_state_data)

            m_sd_df_wb_items = self.create_m_sd_wb_items(
                state_df=state_df,
                state_id_col=self.esm_id_col_dct[dataset]
            )
            # m_sd_df_wb_scales = self.create_m_sd_crits(full_df, dataset=dataset)
            wp_corr, bp_corr, icc1, icc2 = self.calc_bp_wp_statistics(
                df=state_df,
                id_col=self.esm_id_col_dct[dataset],
                tp_col=self.esm_tp_col_dct[dataset],
            )

            if self.desc_cfg["store"]:
                file_path = os.path.join(self.desc_results_base_path, dataset)
                os.makedirs(file_path, exist_ok=True)

                wp_corr = self.format_columns(df=wp_corr, decimals=2)
                file_name_wp_corr = os.path.join(file_path, "within_person_corr_crit.xlsx")
                wp_corr.to_excel(file_name_wp_corr, index=False)

                bp_corr = self.format_columns(df=wp_corr, decimals=2)
                file_name_bp_corr = os.path.join(file_path, "between_person_corr_crit.xlsx")
                bp_corr.to_excel(file_name_bp_corr, index=False)

                file_name_icc1 = os.path.join(file_path, "between_person_corr_crit.xlsx")
                with open(file_name_icc1, 'w') as f:
                    json.dump(icc1, f, indent=4)

                file_name_icc2 = os.path.join(file_path, "between_person_corr_crit.xlsx")
                with open(file_name_icc2, 'w') as f:
                    json.dump(icc2, f, indent=4)

                file_name_wb_items = os.path.join(file_path, "m_sd_wb_items.xlsx")
                m_sd_df_wb_items.to_excel(file_name_wb_items, index=False)

    def create_m_sd_wb_items(self, state_df, state_id_col):
        """

        Returns:

        """
        # TODO Check if CoCoUT is already transformed
        state_df = state_df.drop(state_id_col, axis=1, errors="ignore")
        state_df = state_df.select_dtypes(include=[np.number])
        means = state_df.mean()
        stds = state_df.std()

        # Create a new DataFrame with the means and standard deviations
        m_sd_df = pd.DataFrame({
            'M': means,
            'SD': stds
        })

        # Reset the index for proper formatting and return the DataFrame
        m_sd_df = m_sd_df.reset_index().rename(columns={'index': 'Variable'})

        m_sd_df = self.format_columns(m_sd_df, decimals=2)

        return m_sd_df

    def create_m_sd_crits(self, full_data, dataset):
        """

        Args:
            full_data:
            dataset:

        Returns:

        """
        crits_avail = [f"crit_{key}" for key, datasets in self.var_cfg["analysis"]["crit_available"].items() if dataset in datasets]
        df_crit = full_data.loc[full_data.index.str.startswith("dataset"), crits_avail]
        means = df_crit.mean()
        stds = df_crit.std()

        # Create a new DataFrame with the means and standard deviations
        m_sd_df = pd.DataFrame({
            'M': means,
            'SD': stds
        })

        # Reset the index for proper formatting and return the DataFrame
        m_sd_df = m_sd_df.reset_index().rename(columns={'index': 'Variable'})

        m_sd_df = self.format_columns(m_sd_df, decimals=2)

        return m_sd_df


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
        ICC1_dict = {}
        ICC2_dict = {}

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
            ICC1_dict[var] = ICC1
            ICC2_dict[var] = ICC2

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

        return corr_W, corr_B, ICC1_dict, ICC2_dict

    def compute_rel(self):
        """
        This function computes the reliability of the well-being measures

        Returns:

        """
        pass
        """
        # TODO: Make this sample specific!
        self.datasets
        trait_df_dct, state_df_dct = self.prepare_rel_data()
        # path_to_full_data = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], "full_data")
        # full_df = self.data_loader.read_pkl(path_to_full_data)
        # crit_cols = [col[5:] for col in full_df.columns if col.startswith('crit')]  # remove prefix

        # rel_dct = {} # TODO fix 
        for dataset in self.datasets:
            for crit in crit_cols:
                if "trait" in crit:
                    continue  # TODO ADD
                elif "state" in crit:
                    rel = self.compute_split_half_rel(df_dct=state_df_dct, construct=crit)
                    rel_dct[crit] = rel
                    print(f"Rel {crit}: {rel}")
                else:
                    raise ValueError("Unknown criterium ")
    
            if self.desc_cfg["store"]:
                file_name_rel = os.path.join(self.desc_results_base_path, "weighted_rels.json")
                with open(file_name_rel, 'w') as f:
                    json.dump(rel_dct, f, indent=4)

        return None
        """

    def prepare_rel_data(self):
        """
        This function prepares the data to compute the reliability.

        Returns:
            tuple(pd.DataFrame, dict): trait and state df processed for reliability analysis
        """
        # state data
        state_dfs = {}
        for dataset in self.datasets:
            timestamp_col = self.var_cfg["preprocessing"]["esm_timestamp_col"][dataset]
            id_col = self.var_cfg["preprocessing"]["esm_id_col"][dataset]

            path_to_state_data = os.path.join(self.data_base_path, f"wb_items_{dataset}")
            state_df = self.data_loader.read_pkl(path_to_state_data)

            # Ensure timestamp is in datetime format
            state_df[timestamp_col] = pd.to_datetime(state_df[timestamp_col])

            # Create idx_measurement_per_person
            state_df['idx_measurement_per_person'] = state_df.sort_values(by=timestamp_col).groupby(id_col).cumcount()
            state_df["joint_user_id"] = state_df[id_col].apply(lambda x: f"{dataset}_{x}")
            state_df = state_df.sort_values([id_col, "idx_measurement_per_person"])
            columns_to_select = ["pa_state", "na_state", "wb_state", "idx_measurement_per_person", "joint_user_id"]
            state_dfs[dataset] = state_df[state_df.columns.intersection(columns_to_select)]

        # trait data
        trait_dfs = {}
        for dataset in self.datasets:
            path_to_trait_data = os.path.join(self.data_base_path, f"trait_wb_items_{dataset}")
            trait_df = self.data_loader.read_pkl(path_to_trait_data)
            trait_dfs[dataset] = trait_df

        return trait_dfs, state_dfs

    def compute_internal_consistency(self,
                                     df_items,
                                     ):
        """
        This function computes the internal consistency as a reliability measure of the trait wb_items

        Returns:

        """
        alpha = None
        if len(df_items.columns) > 1:
            alpha = pg.cronbach_alpha(data=df_items)[0]
        return alpha




    def compute_split_half_rel(self,
                               df_dct: dict[pd.DataFrame],
                               construct: str,
                               user_id_col: str = "joint_user_id",
                               measurement_idx_col: str = "idx_measurement_per_person",
                               method: str = "individual_halves",
                               correlation_method: str = "spearman") -> float:
        """  # TODO: This should currently be per sample
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
        rel_dct = {}

        for dataset, df in df_dct.items():
            # Check if data is present
            if construct not in df.columns:  # This may happen
                print(f"The construct '{construct}' is not a column in the DataFrame for dataset {dataset}.")
                continue
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

            rel_dct[dataset] = reliability
            print(f"Reliability for dataset {dataset}: {np.round(reliability, 3)}")

        weighted_mean_rel = self.compute_weighted_mean_rel(df_dct=df_dct, rel_dct=rel_dct, user_id_col=user_id_col)
        print(f"Weighted mean reliability for '{construct}' across datasets: {np.round(weighted_mean_rel, 3)}")

        return weighted_mean_rel

    @staticmethod
    def compute_weighted_mean_rel(df_dct: dict, rel_dct: dict, user_id_col: str) -> float:
        """
        This function computes the mean reliability across samples, weighted by sample size

        Args:
            df_dct:
            rel_dct:
            user_id_col:

        Returns:
            float: weighted rel

        """
        # Calculate the weighted mean reliability
        total_unique_users = 0
        weighted_sum_reliability = 0

        for dataset, reliability in rel_dct.items():
            # Count unique users in the current dataset
            unique_user_count = df_dct[dataset][user_id_col].nunique()
            # Accumulate weighted reliability and user counts
            weighted_sum_reliability += reliability * unique_user_count
            total_unique_users += unique_user_count
        weighted_mean_rel = weighted_sum_reliability / total_unique_users if total_unique_users > 0 else np.nan

        return weighted_mean_rel














