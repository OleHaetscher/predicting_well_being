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
