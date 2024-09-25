import re
from copy import deepcopy

import numpy as np

from src.preprocessing.BasePreprocessor import BasePreprocessor
import pandas as pd


class EmotionsPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg, var_cfg)
        self.dataset = "emotions"
        self.close_interactions = None

    def merge_traits(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges DataFrames from the input dictionary that have 'traits' in their key.

        Only columns that have the same value across all rows for a given 'id_for_merging' are retained.

        Args:
            df_dct (dict[str, pd.DataFrame]): A dictionary where the keys are strings and
                                              the values are pandas DataFrames. DataFrames
                                              containing 'traits' in their key are considered for merging.

        Returns:
            pd.DataFrame: A single DataFrame containing the concatenated 'traits' DataFrames,
                          with columns filtered to include only those that do not vary across
                          rows with the same 'id_for_merging'.
        """
        trait_esm_df = df_dct["data_traits_esm"]
        df_traits = trait_esm_df.drop_duplicates(subset=self.raw_trait_id_col, keep="first").reset_index(drop=True)
        # df_traits = self.split_trait_ids(df_traits=df_traits)
        return df_traits

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:  # Report in PreReg
        """
        Removes specified suffixes from all column names in the DataFrame if the suffixes are present.
        Fills missing values in the base column (e.g., 'narqs_1') with values from the corresponding
        higher-suffix columns (e.g., '_t2', '_t3', '_t4'). If there are NaN values in '_t2', it looks
        for a value in '_t3', and so on.

        If a base column does not exist, the first available suffix column (e.g., '_t2') will be renamed
        as the base column after filling NaNs.

        Args:
            df_traits (pd.DataFrame): A pandas DataFrame whose column names need to be updated.

        Returns:
            pd.DataFrame: A pandas DataFrame with updated column names and values, ensuring no duplicates.
        """
        # TODO: This is elegant -> use as general method?
        trait_suffixes = ["_t2", "_t3", "_t4"]  # Reversed order for filling NaNs from higher to lower
        suffix_pattern = re.compile(r"_t\d$")

        # Process each column to either fill NaNs or rename
        for col in df_traits.columns:
            # Extract the base column (without any suffix)
            base_col = re.sub(suffix_pattern, '', col)

            # If the column is a base column (i.e., does not end with any of the suffixes)
            # if base_col in df_traits.columns:
                # Process only the base column for NaN filling
            if base_col not in df_traits.columns:
                df_traits[base_col] = np.nan
            for suffix in trait_suffixes:  # Start with '_t2', then '_t3', and '_t4'
                suffix_col = f"{base_col}{suffix}"
                if suffix_col in df_traits.columns:
                    # Fill NaNs in the base column with the suffix column values
                    df_traits[base_col].fillna(df_traits[suffix_col], inplace=True)
                    df_traits = df_traits.drop(columns=suffix_col)

        assert len(df_traits.columns) == len(set(df_traits.columns)), "Duplicate column names found after renaming!"

        return df_traits

    def merge_states(self, df_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges DataFrames from the input dictionary that have 'states' in their key.

        Only columns that have different values across rows for a given 'id_for_merging' are retained.

        Args:
            df_dct (dict[str, pd.DataFrame]): A dictionary where the keys are strings and
                                              the values are pandas DataFrames. DataFrames
                                              containing 'states' in their key are considered for merging.

        Returns:
            pd.DataFrame: A single DataFrame containing the concatenated 'states' DataFrames,
                          with columns filtered to include only those that vary across
                          rows with the same 'id_for_merging'.
        """
        state_df = df_dct["data_traits_esm"]
        # Filter columns to include only those that vary by 'id_for_merging'
        grouped = state_df.groupby(self.raw_esm_id_col)
        varying_columns = [col for col in state_df.columns
                           if grouped[col].nunique().gt(1).any()]
        state_df_filtered = state_df[varying_columns + [self.raw_esm_id_col] + ["wave"]]
        # state_df_filtered = self.split_state_ids(state_df_filtered)
        return state_df_filtered

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases.

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        df_states = self.create_close_interactions(df_states=df_states)
        df_states = self.merge_int_occup_states(df_states=df_states)
        return df_states

    def create_close_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In Emotions, we need another custom logic. With each prompt, we refer to 5 interaction partners
        (interaction_partner1 - interaction_partner5). We only consider an interaction close tie,
        if all interaction partners are close ties (and weak ties, if all interaction partners are weak ties,
        respectively).

        Args:
            df_states:

        Returns:
            df_states
        """
        close_interaction_cfg = [entry for entry in self.fix_cfg["esm_based"]["self_reported_micro_context"]
                                 if entry["name"] == "close_interactions"][0]
        int_partner_cols = close_interaction_cfg["special_mappings"]["emotions"]["columns"]
        cat_mapping = close_interaction_cfg["special_mappings"]["emotions"]["mapping"]

        # Apply the categorical mapping to each interaction partner column
        for col in int_partner_cols:
            df_states[col] = df_states[col].replace(cat_mapping)

        # Define the close and weak tie logic (either 1 or NaN, or 0 or NaN)
        all_close_mask = df_states[int_partner_cols].isin([1, np.nan]).all(axis=1)
        all_weak_mask = df_states[int_partner_cols].isin([0, np.nan]).all(axis=1)

        # Create the "close_interactions_raw" column
        df_states['close_interactions_raw'] = np.where(all_close_mask, 1,
                                                       np.where(all_weak_mask, 0, np.nan))

        # Calculate the percentage of close ties per person (grouped by unique_id)
        interaction_stats = df_states.groupby(self.raw_esm_id_col)['close_interactions_raw'].apply(
            lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
        )

        # Create the "close_interactions" column
        df_states['close_interactions'] = df_states[self.raw_esm_id_col].map(interaction_stats)
        self.close_interactions = deepcopy(df_states[["close_interactions", self.raw_esm_id_col]].drop_duplicates(keep="first"))

        return df_states

    def merge_int_occup_states(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In emotions, we have different columns for state_pa and state_na, depending on whether an
        interaction took place or not. This could bias the pl summaries of state affect.
        Therefore, we set one of the cols to np.nan and fill the other column with the missing values.
        Then we do not have to modify anything in the config.

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        crit_state_cfg = self.fix_cfg["esm_based"]["criterion"]
        pa_items = crit_state_cfg[0]["item_names"][self.dataset]
        na_items = crit_state_cfg[1]["item_names"][self.dataset]  # Assuming the second part for NA items

        # Process both PA and NA items
        for item in pa_items + na_items:
            # Separate into 'occup' and 'int' columns based on the naming convention
            occup_col = f'occup_{item.split("_")[1]}'  # E.g., "occup_relaxed"
            int_col = f'int_{item.split("_")[1]}'  # E.g., "int_relaxed"

            if occup_col in df_states.columns and int_col in df_states.columns:
                # Merge the two columns: if 'occup' is NaN, fill it with 'int', and vice versa
                df_states[occup_col] = df_states[occup_col].combine_first(df_states[int_col])
                df_states[int_col] = df_states[occup_col] # same values

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        No custom adjustments necessary in cocoesm.

        Args:
            df:

        Returns:
            pd.DataFrame
        """
        df = df.merge(self.close_interactions, on=self.raw_esm_id_col, how="left")
        return df

    def split_trait_ids(self, df_traits: pd.DataFrame) -> pd.DataFrame:  # Not used ATM
        """
        We use this function to split the ids for people, that participated in both waves, otherwise
        the calculations of some features become unreasonable. Therefore, each ID that participated
        in both Waves is split into to separate rows (e.g., if the original id is 30, the resulting
        IDs will be 30_1, 30_2). One row gets the data from the first wave (i.e., all columns with no suffix
        and the suffix _t2) and the other row gets data from the second wave (i.e., all columns with
        the suffixes _t3 and _t4).

        Args:
            df_traits (pd.DataFrame): DataFrame with participant data, including a 'wave' column.

        Returns:
            pd.DataFrame: DataFrame with IDs split by wave.
        """
        df_both = df_traits[df_traits['wave'] == 'Both'].copy()
        df_wave1 = df_both.copy()
        df_wave2 = df_both.copy()

        df_wave1[self.raw_trait_id_col] = df_wave1[self.raw_trait_id_col].astype(str) + '_1'
        df_wave2[self.raw_trait_id_col] = df_wave2[self.raw_trait_id_col].astype(str) + '_2'

        wave1_cols = [col for col in df_traits.columns if not col.endswith(('_t3', '_t4'))]
        wave2_cols = [self.raw_trait_id_col] + [col for col in df_traits.columns if col.endswith(('_t3', '_t4'))]

        df_wave1 = df_wave1[wave1_cols]
        df_wave2 = df_wave2[wave2_cols]
        joined_df = pd.concat([df_wave1, df_wave2, df_traits[df_traits['wave'] != 'Both']], ignore_index=True)

        return joined_df

    def split_state_ids(self, df_states: pd.DataFrame) -> pd.DataFrame:  # Not used ATM
        """
        Splits state IDs by appending "_1" to the self.raw_esm_id_col if the 'dataset' column value is "S2W1",
        and "_2" if the value is "S2W2", but only for participants who have both "S2W1" and "S2W2"
        in the dataset.

        Args:
            df_states (pd.DataFrame): DataFrame containing participant states, including the 'dataset' column.

        Returns:
            pd.DataFrame: DataFrame with updated self.raw_esm_id_col for eligible participants.
        """
        # Identify participants with both "S2W1" and "S2W2" in the dataset
        ids_with_both = df_states.groupby(self.raw_esm_id_col)['dataset'].apply(lambda x: {'S2W1', 'S2W2'}.issubset(set(x)))

        # Filter for those participants
        eligible_ids = ids_with_both[ids_with_both].index

        # Apply the split only to eligible participants
        df_states[self.raw_esm_id_col] = df_states.apply(
            lambda row: f"{row[self.raw_esm_id_col]}_1" if row[self.raw_esm_id_col] in eligible_ids and row['dataset'] == 'S2W1'
            else f"{row[self.raw_esm_id_col]}_2" if row[self.raw_esm_id_col] in eligible_ids and row['dataset'] == 'S2W2'
            else row[self.raw_esm_id_col], axis=1
        )

        return df_states
