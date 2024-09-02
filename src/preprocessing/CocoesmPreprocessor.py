from src.preprocessing.BasePreprocessor import BasePreprocessor
import pandas as pd
import re


class CocoesmPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg=fix_cfg, var_cfg=var_cfg)
        self.dataset = "cocoesm"

    def merge_traits(self, df_dct):
        return df_dct["data_traits"]

    def merge_states(self, df_dct):
        data_esm = df_dct["data_esm"]
        data_esm_daily = df_dct["data_esm_daily"]
        data_esm['date'] = pd.to_datetime(data_esm['created_individual']).dt.date
        data_esm_daily['date'] = pd.to_datetime(data_esm_daily['created_individual']).dt.date
        # Merge the DataFrames based on 'participant' and 'date' columns
        merged_df = pd.merge(
            data_esm,
            data_esm_daily,
            how='left',
            on=['participant', 'date'],
            suffixes=("_esm", "_daily"),
        )
        return merged_df

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Removes a specified suffix from all column names in the DataFrame if the suffix is present.
        Additionally, removes the 'r' in column names that match a regex pattern of a number followed by 'r'.

        Args:
            df_traits: A pandas DataFrame whose column names need to be updated.
            suffix: A string suffix to be removed from the column names. Default is '_t1'.
            regex_pattern: A regex pattern to match and remove the 'r' after a number. Default is r'(\d)r$'.

        Returns:
            A pandas DataFrame with the updated column names.
        """
        trait_suffix = "_t1"
        regex_pattern: str = r'(\d)r$'
        updated_columns = []
        for col in df_traits.columns:
            # Remove suffix if present
            if col.endswith(trait_suffix):
                col = col[:-len(trait_suffix)]
            # Remove 'r' from columns matching the regex pattern
            col = re.sub(regex_pattern, r'\1', col)
            updated_columns.append(col)
        df_traits.columns = updated_columns
        return df_traits

    def dataset_specific_esm_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        No custom adjustments necessary in cocoesm.

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        return df_states