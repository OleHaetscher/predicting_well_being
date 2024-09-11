from src.preprocessing.BasePreprocessor import BasePreprocessor
import pandas as pd

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

    def merge_states(self, df_dct):
        return df_dct["data_esm"]

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates variable names corresponding to cfg_fix (i.e., it removes the timepoint suffixes
        from the column names and deletes the duplicate columns [e.g., _t1, _t2). We chose to use the earliest
        trait survey data in each dataset. Because of the data particularities, this may be implemented in the
        subclasses (if there is some kind of suffix behind the var name defined in cfg_fix.

        Args:
            df_traits: A pandas DataFrame containing the dataset.

        Returns:
            pd.DataFrame: A DataFrame with clean columns names
        """
        return df_traits


