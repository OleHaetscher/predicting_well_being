import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor


class ZpidPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg, var_cfg)
        self.dataset = "zpid"

    def merge_traits(self, df_dct):
        return df_dct["data_traits"]

    def merge_states(self, df_dct):
        return df_dct["data_esm"]

    def dataset_specific_trait_processing(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        In ZPID, we need to
            - set the numerical values that mean "not specified" to np.nan
            Polit_Ein_4: 12; Demo_GL2: 5;

        Args:
            df_traits:

        Returns:
            pd.DataFrame:
        """
        df_traits = self.replace_values(df_traits=df_traits)
        df_traits = self.merge_bfi_items(df_traits=df_traits)
        return df_traits

    def replace_values(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method replaces certain values that
          - are equivalent to np.nan (e.g., -77)
          - are equivalent to "I do not want to answer this"

        Args:
            df_traits:

        Returns:

        """
        df_traits = df_traits.replace(-77, np.nan)
        df_traits.loc[df_traits['Demo_GL2'] == 5, 'Demo_GL2'] = np.nan
        df_traits.loc[df_traits['Polit_Ein_4'] == 12, 'Polit_Ein_4'] = np.nan
        return df_traits

    def merge_bfi_items(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method merges the bfi2xs items assessed in two waves to compensate missings and increase reliability
          - If a person has values on an item in both waves, we calculate the mean
          - If a person has only a value on one of the item, we take that value
          - If a person has no values, the resulting value wil be set to np.nan

        Args:
            df_traits:

        Returns:
            pd.DataFrame:
        """
        # Identify all BFI_2XS items
        bfi_items = set([col.rsplit('_', 1)[0] for col in df_traits.columns if "BFI_2XS" in col])

        # For each BFI item, merge wave 3 and wave 4
        for item in bfi_items:
            wave3_col = f"{item}_wave3"
            wave4_col = f"{item}_wave4"

            # Merge wave 3 and wave 4 by taking the mean, with skipna=True to ignore NaNs
            df_traits[item] = df_traits[[wave3_col, wave4_col]].mean(axis=1, skipna=True)

        # Drop the original wave-specific columns
        df_traits.drop(columns=[f"{item}_wave3" for item in bfi_items] + [f"{item}_wave4" for item in bfi_items],
                       inplace=True)

        return df_traits










