import numpy as np
import pandas as pd

from src.preprocessing.BasePreprocessor import BasePreprocessor


class CocomsPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg, var_cfg)
        self.dataset = "cocoms"

    def merge_traits(self, df_dct):
        traits_dfs = [df for key, df in df_dct.items() if 'traits' in key]
        concatenated_traits = pd.concat(traits_dfs, axis=0).reset_index(drop=True)
        return concatenated_traits

    def merge_states(self, df_dct):
        esm_dfs = [df for key, df in df_dct.items() if 'esm' in key]
        concatenated_esm = pd.concat(esm_dfs, axis=0).reset_index(drop=True)
        return concatenated_esm

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Removes a specified suffix from all column names in the DataFrame if the suffix is present.
        Additionally, removes the 'r' in column names that match a regex pattern of a number followed by 'r'.

        Args:
            df_traits: A pandas DataFrame whose column names need to be updated.

        Returns:
            A pandas DataFrame with the updated column names.
        """
        trait_suffix = "_t1"
        updated_columns = []
        for col in df_traits.columns:
            if col.endswith(trait_suffix):
                col = col[:-len(trait_suffix)]
            updated_columns.append(col)
        df_traits.columns = updated_columns
        return df_traits

    def dataset_specific_trait_processing(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        In CoCoMS, we need to
            - assign numerical values to the elected political parties and strech this scale to 1-11

        Args:
            df_traits:

        Returns:
            pd.DataFrame:
        """
        party_number_map = [entry["party_num_mapping"] for entry in self.fix_cfg["predictors"]["person_level"]["personality"]
                            if "party_num_mapping" in entry.keys()][0]["cocoms"]
        df_traits['political_orientation'] = df_traits['vote_general'].map(party_number_map).fillna(np.nan)
        return df_traits



    def dataset_specific_esm_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method must be implemented in the subclasses. It describes some dataset specific processing
        to handle dataset-specific differences in the variables or scales.

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        pass

    def align_interaction_partner_cols(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df_states:

        Returns:

        """


