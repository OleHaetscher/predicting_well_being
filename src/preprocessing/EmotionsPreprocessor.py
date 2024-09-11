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
        trait_df = df_dct["data_traits_esm"]

        # Filter columns to include only those that do not vary by 'id_for_merging'
        grouped = trait_df.groupby('id')   # TODO "id" in old data, "id_for_merging" in new, incomplete data
        consistent_columns = [col for col in trait_df.columns
                              if grouped[col].nunique().eq(1).all()]
        collapsed_df = grouped.first().reset_index()
        only_trait_df = collapsed_df[consistent_columns]
        return only_trait_df

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the '_t3' suffix from all column names in the DataFrame,
        except when doing so would result in duplicate column names.

        Args:
            df_traits (pd.DataFrame): A pandas DataFrame whose column names need to be updated.

        Returns:
            pd.DataFrame: A pandas DataFrame with updated column names, ensuring no duplicates are created.
        """
        trait_suffix = "_t3"
        updated_columns = []
        seen_columns = set()

        for col in df_traits.columns:
            new_col = col
            if col.endswith(trait_suffix):
                new_col = col[:-len(trait_suffix)]

            # Check for potential duplicates
            if new_col not in seen_columns:
                seen_columns.add(new_col)
                updated_columns.append(new_col)
            else:
                updated_columns.append(col)

        df_traits.columns = updated_columns
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
        grouped = state_df.groupby('id')  # TODO "id" in old data, "id_for_merging" in new, incomplete data
        varying_columns = [col for col in state_df.columns
                           if grouped[col].nunique().gt(1).any()]
        state_df_filtered = state_df[varying_columns]
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
        close_interaction_cfg = [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
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
        interaction_stats = df_states.groupby('unique_id')['close_interactions_raw'].apply(
            lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
        )

        # Create the "close_interactions" column
        df_states['close_interactions'] = df_states['unique_id'].map(interaction_stats)

        return df_states



