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
        trait_df = trait_esm_df.drop_duplicates(subset=self.raw_trait_id_col, keep="first").reset_index(drop=True)
        return trait_df

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
        state_df_filtered = state_df[varying_columns + [self.raw_esm_id_col]]
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

        # Define the close and weak tie logic
        all_close_mask = df_states[int_partner_cols].eq(1).all(axis=1)  # All columns are close ties (1)
        all_weak_mask = df_states[int_partner_cols].eq(0).all(axis=1)  # All columns are weak ties (0)

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



