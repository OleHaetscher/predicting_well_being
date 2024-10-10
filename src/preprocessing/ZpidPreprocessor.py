from copy import deepcopy
from datetime import datetime

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
        self.home_office = None

    def merge_traits(self, df_dct):
        n = df_dct["data_traits"]["p_0001"].nunique()
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
        self.logger.log(f" ")
        df_traits = self.replace_values(df_traits=df_traits)
        df_traits = self.merge_bfi_items(df_traits=df_traits)
        df_traits = self.adjust_professional_status_col(df_traits=df_traits)
        df_traits = self.create_home_office(df_traits=df_traits)
        df_traits["created_demog"] = "2020-01-01"
        df_traits["country"] = "germany"
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
        df_traits.loc[df_traits['Demo_B1'] == 0, 'Demo_B1'] = np.nan
        df_traits.loc[df_traits['Arbeitsleben91_AugOct'] == 0, 'Arbeitsleben91_AugOct'] = np.nan
        df_traits.loc[df_traits['Demo_AL1'] == 0, 'Demo_AL1'] = np.nan

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

    def adjust_professional_status_col(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the professional status column "Demo_AL1". It changes the categories 1 (Angestellter) and
        3 (Beamter) to new category 15 if the number of days worked per week in the "MCTQ_WD_2" column
        is less than 5.

        Args:
            df_traits (pd.DataFrame): A pandas DataFrame containing the columns "Demo_AL1" (professional status)
                                      and "MCTQ_WD_2" (number of days worked per week).

        Returns:
            pd.DataFrame: The updated DataFrame with modified "Demo_AL1" values where applicable.
        """

        # Define the condition where "MCTQ_WD_2" is less than 5 and "Demo_AL1" is either 1 or 3
        condition = (df_traits["MCTQ_WD_2"] < 5) & df_traits["Demo_AL1"].isin([1, 3])

        # Update "Demo_AL1" to 15 where the condition is true
        df_traits.loc[condition, "Demo_AL1"] = 14

        return df_traits

    def create_home_office(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates

        Args:
            df_traits:

        Returns:

        """
        home_office_cfg = self.config_parser(self.fix_cfg["person_level"]["sociodemographics"],
                                           "percentage",
                                           "home_office")[0]
        col_name = home_office_cfg["item_names"][self.dataset]
        df_traits[col_name] = df_traits[col_name].map(home_office_cfg["category_mappings"][self.dataset])
        df_traits["home_office"] = df_traits[col_name] / 7  # 7 days would be 100%
        self.home_office = deepcopy(df_traits[["home_office", self.raw_trait_id_col]].drop_duplicates(keep="first"))
        return df_traits

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In ZPID, we need to
            - create a column that tracks if a particapant participated in both ESM-surveys

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        df_states = self.create_wave_col(df_states)
        return df_states

    def create_wave_col(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates a column 'studyWave' that tracks in which study waves a person participated.
        Wave 1 includes timestamps before the end of August, and Wave 2 includes timestamps from September onwards.
        The result will be:
        - 1: Only participated in Wave 1
        - 2: Only participated in Wave 2
        - 3: Participated in both waves

        Args:
            df_states: DataFrame containing ESM data with a timestamp column.

        Returns:
            pd.DataFrame: Updated DataFrame with a new 'wave' column.
        """
        # Define the wave cutoff dates
        wave_1_end = datetime(year=2020, month=8, day=31)
        wave_2_start = datetime(year=2020, month=9, day=1)
        df_states[self.esm_timestamp] = pd.to_datetime(df_states[self.esm_timestamp])

        # Group by person_id and calculate the first and last timestamps
        grouped = df_states.groupby(self.raw_esm_id_col)[self.esm_timestamp].agg(['min', 'max']).reset_index()

        # Function to assign waves based on the first and last timestamps
        def assign_wave(row):
            first_timestamp = row['min']
            last_timestamp = row['max']

            participated_in_wave_1 = first_timestamp <= wave_1_end
            participated_in_wave_2 = last_timestamp >= wave_2_start

            if participated_in_wave_1 and participated_in_wave_2:
                return "Both"  # Both waves
            elif participated_in_wave_1:
                return 1  # Only wave 1
            elif participated_in_wave_2:
                return 2  # Only wave 2
            return np.nan  # No valid participation (shouldn't occur)

        # Assign the wave to each person
        grouped['studyWave'] = grouped.apply(assign_wave, axis=1)

        # Merge the wave data back into the original dataframe
        df_states = df_states.merge(grouped[[self.raw_esm_id_col, 'studyWave']], on=self.raw_esm_id_col, how='left')

        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            df:

        Returns:
            pd.DataFrame
        """
        df = df.merge(self.home_office, on=self.raw_trait_id_col, how="left")
        return df

    def dataset_specific_sensing_processing(self, df_sensing: pd.DataFrame) -> pd.DataFrame:
        """
        Overridden in the subclasses

        Args:
            df_sensing:

        Returns:

        """
        df_sensing = self.fill_nans_in_app_features(df_sensing=df_sensing)
        # TODO: This may be adjusted to differ true nan from 0
        return df_sensing

    def fill_nans_in_app_features(self, df_sensing: pd.DataFrame) -> pd.DataFrame:
        """
        In the features that contain durations of App usage, there are a lot of np.nan values in the sensing df
        (which means that persons have not installed an app of that category at all, e.g., "spirituality_apps")
        In this method, we set these values to zero (i.e., because they did not use the app, if they do not have the
        app on the phone at all)

        Args:
            df_sensing:

        Returns:
            pd.DataFrame
        """
        app_cols = [col for col in df_sensing.columns if "app_" in col]
        df_sensing[app_cols] = df_sensing[app_cols].fillna(0)
        return df_sensing










