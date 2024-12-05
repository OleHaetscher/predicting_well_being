import os
import numpy as np
import pandas as pd
from src.utils.DataLoader import DataLoader
import openpyxl


class DescriptiveStatistics:
    """
    This class computes descriptives as specified in the PreReg. This includes
        - M and SD of all features used in the ML-based analysis
        - means, standard deviations, the proportion of between-person variance (ICC),
            and within-person and between-person correlations for the wb_items and the
            pa, na, and wb-scores
    """
    def __init__(self, fix_cfg, var_cfg, name_mapping):
        self.var_cfg = var_cfg
        self.data_loader = DataLoader()
        self.fix_cfg = fix_cfg
        self.name_mapping = name_mapping
        self.esm_id_col_dct = self.fix_cfg["esm_based"]["other_esm_columns"][0]["item_names"]

        self.full_data_path = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], "full_data")
        self.state_data_base_path = self.var_cfg['analysis']["path_to_preprocessed_data"]

    def create_m_sd_feature_table(self):
        """
        This method creates a table containing the mean (M) and standard deviation (SD) for continuous variables,
        and frequencies (counts and percentages) for binary variables. The table is subdivided based on the name
        of the features (visible by the prefix of the column names: 'pl_', 'sens_', 'srmc_', 'mac_').

        Returns:
            final_table: pd.DataFrame containing the formatted table in APA style.
        """
        # TODO: Per Sample or in total? Currently total
        # path_to_full_data = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], "full_data")
        full_df = self.data_loader.read_pkl(self.full_data_path)
        results = []
        feature_cats = ['pl', 'srmc', 'sens', 'mac']

        for cat in feature_cats:
            prefix = f"{cat}_"
            prefixed_cols = [col for col in full_df.columns if col.startswith(prefix)]
            if not prefixed_cols:
                continue
            df_subset = full_df[prefixed_cols]

            binary_vars, continuous_vars = self.separate_binary_continuous_cols(df_subset)

            # Compute statistics for continuous variables
            if continuous_vars:
                cont_stats = self.calculate_cont_stats(df=df_subset, continuous_vars=continuous_vars, prefix=prefix)
                results.append(cont_stats)

            # Compute frequencies for binary variables
            if binary_vars:
                bin_stats = self.calculate_bin_stats(df=df_subset, binary_vars=binary_vars, prefix=prefix)
                results.append(bin_stats)

        # Combine all results
        final_table = pd.concat(results, ignore_index=True)

        # Format numerical columns to two decimal places  # TODO make general method?
        final_table['M'] = final_table.get('M', pd.Series()).round(2)
        final_table['SD'] = final_table.get('SD', pd.Series()).round(2)
        final_table['Percentage'] = final_table.get('Percentage', pd.Series()).round(2)

        # Rename feature names
        final_table["Group"] = final_table["Group"].replace(self.name_mapping["category_names"])
        for cat in feature_cats:
            final_table['Variable'] = final_table['Variable'].replace(self.name_mapping[cat])

        final_table = final_table.reset_index(drop=True)
        final_table.to_excel("test.xlsx", index=False)
        print(final_table.to_string(index=False))

    @staticmethod
    def calculate_cont_stats(df, continuous_vars, prefix):
        """

        Args:
            df:
            continuous_vars:
            prefix:

        Returns:

        """
        cont_stats = df[continuous_vars].agg(['mean', 'std']).transpose().reset_index()
        cont_stats.columns = ['Variable', 'M', 'SD']
        cont_stats['Variable'] = cont_stats['Variable'].str.replace(prefix, '', regex=False)
        cont_stats['Group'] = prefix.rstrip('_')
        # Reorder columns for APA style
        cont_stats = cont_stats[['Group', 'Variable', 'M', 'SD']]
        return cont_stats

    @staticmethod
    def calculate_bin_stats(df, binary_vars, prefix):
        """

        Args:
            df:
            binary_vars:
            prefix:

        Returns:

        """
        bin_stats = df[binary_vars].apply(lambda x: x.value_counts(dropna=True)).transpose()

        # Clean variable names (remove prefix)
        bin_stats['Variable'] = bin_stats.index.str.replace(prefix, '', regex=False)

        # Initialize the 'Frequency' column
        bin_stats['Frequency'] = np.nan

        # Handle counts for '1' (could be 1 or 1.0)
        if 1 in bin_stats.columns:
            bin_stats['Frequency'] = bin_stats[1]
        elif 1.0 in bin_stats.columns:
            bin_stats['Frequency'] = bin_stats[1.0]
        # If neither '1' nor '1.0' is present, 'Frequency' remains NaN

        # Calculate the total count excluding NaN values for each variable
        bin_stats['Total'] = df[binary_vars].notna().sum().values

        # Calculate percentage of '1' occurrences, handling NaN frequencies properly
        bin_stats['Percentage'] = (bin_stats['Frequency'] / bin_stats['Total']) * 100

        # Add the group name (derived from the prefix)
        bin_stats['Group'] = prefix.rstrip('_')

        # Select relevant columns and reset index
        bin_stats = bin_stats[['Group', 'Variable', 'Frequency', 'Total', 'Percentage']].reset_index(drop=True)

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

    def create_wb_item_statistics(self):
        """

        Returns:

        """
        for dataset in self.var_cfg["general"]["datasets_to_be_included"]:
            path_to_full_data = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], f"wb_items_{dataset}")
            state_df = self.data_loader.read_pkl(path_to_full_data)
            m_sd_df = self.create_m_sd_wb_items(state_df)
            wp_corr, bp_corr, icc1, icc2 = self.calc_bp_wp_statistics(state_df, self.esm_id_col_dct[dataset])
            print()

    @staticmethod
    def create_m_sd_wb_items(state_df):
        """

        Returns:

        """
        means = state_df.mean()
        stds = state_df.std()

        # Create a new DataFrame with the means and standard deviations
        m_sd_df = pd.DataFrame({
            'M': means,
            'SD': stds
        })

        # Reset the index for proper formatting and return the DataFrame
        m_sd_df = m_sd_df.reset_index().rename(columns={'index': 'Variable'})

        return m_sd_df

    @staticmethod
    def calc_bp_wp_statistics(df, id_col):
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
        variables_to_correlate = df.columns.drop(id_col)

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
        trait_df, state_df_dct =  self.prepare_rel_data()
        path_to_full_data = os.path.join(self.var_cfg['analysis']["path_to_preprocessed_data"], "full_data")
        full_df = self.data_loader.read_pkl(path_to_full_data)
        crit_cols = [col[5:] for col in full_df.columns if col.startswith('crit')]

        rel_dct = {}
        for crit in crit_cols:
            if "trait" in crit:
                continue  # focus on state data first
            elif "state" in crit:
                rel = self.compute_split_half_rel(df_dct=state_df_dct, construct=crit)
                rel_dct[crit] = rel
                print(f"Rel {crit}: {rel}")
            else:
                raise ValueError("Unknown criterium ")
        return rel_dct["state_wb"]


    def prepare_rel_data(self):
        """

        Returns:
            tuple(pd.DataFrame, dict): trait and state df processed for reliability analysis
        """
        # TODO ADJUST
        # state data
        state_dfs = {}
        for dataset in self.var_cfg["general"]["datasets_to_be_included"]:
            timestamp_col = self.var_cfg["preprocessing"]["esm_timestamp_col"][dataset]
            id_col = self.var_cfg["preprocessing"]["esm_id_col"][dataset]

            path_to_state_data = os.path.join(self.state_data_base_path, f"wb_items_{dataset}")
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
        full_df = self.data_loader.read_pkl(self.full_data_path)
        full_df_filtered = full_df[["crit_wb_trait", "crit_pa_trait", "crit_na_trait"]]
        # TODO: I need single items for trait rel? -> Change preprocessor

        return full_df_filtered, state_dfs

    def compute_split_half_rel(self,
                               df_dct: dict[pd.DataFrame],
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














