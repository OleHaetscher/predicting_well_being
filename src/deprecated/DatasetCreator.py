import pandas as pd
from src.utils.logger import Logger


class DatasetCreator:

    """
    """

    def __init__(self,
                 data_cocoesm: pd.DataFrame,
                 data_cocoms: pd.DataFrame,
                 data_cocout: pd.DataFrame,
                 data_emotions: pd.DataFrame,
                 data_pia: pd.DataFrame,
                 data_zpid: pd.DataFrame,
                 ):
        """
        Initializes the BasePreprocessor with a configuration file.

        Args:
            fix_cfg (dict): yaml config
            var_cfg (dict): yaml config
        """
        self.data_cocoesm = data_cocoesm
        self.data_cocoms = data_cocoms
        self.data_cocout = data_cocout
        self.data_emotions = data_emotions
        self.data_pia = data_pia
        self.data_zpid = data_zpid
        self.logger = Logger()  # composition

    def merge_dfs(self):
        """
        This function merges the different dfs and creates the final dataframe. The index of the final DataFrame is
        a combination of the common id column ("id") and the dataset (e.g., "coco_esm").

        Returns:
            pd.DataFrame

        """

    def store_df(self):
        """
        This function stores the final df as a .pkl file

        Returns:
            .pkl: Pickle file containing the final df
        """
        


