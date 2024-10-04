from src.preprocessing.BasePreprocessor import BasePreprocessor
import pandas as pd
import re
import numpy as np

class PiaPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg, var_cfg)
        self.dataset = "pia"

    def merge_traits(self, df_dct):
        return df_dct["data_traits"]

    def dataset_specific_trait_processing(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df_traits:

        Returns:

        """
        df_traits["country"] = "germany"
        return df_traits

    def merge_states(self, df_dct):
        return df_dct["data_esm"]

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



