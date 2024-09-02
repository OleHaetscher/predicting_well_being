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
        df_traits.loc[df_traits['Demo_GL2'] == 5, 'Demo_GL2'] = np.nan
        df_traits.loc[df_traits['Polit_Ein_4'] == 12, 'Polit_Ein_4'] = np.nan
