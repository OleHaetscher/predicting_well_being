from functools import reduce

import numpy as np

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

    def dataset_specific_trait_processing(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases. In CoCo ESM, this includes
        - Filling NaNs in 'living_other', 'living_partner', and 'living_children' with zero
          if 'living_alone' == 1.

        Args:
            df_traits (pd.DataFrame): The DataFrame containing trait data.

        Returns:
            pd.DataFrame: The modified DataFrame after dataset-specific processing.
        """
        # Define the columns to fill if the condition is met
        df_traits.loc[df_traits['quantity_household'] == 1, "relationship_household"] = '0'
        return df_traits

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        No custom adjustments necessary in cocoesm.

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        df_states = self.create_relationship(df_states=df_states)
        return df_states

    def create_relationship(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Infers the relationship status from the ESM surveys based on interactions with a partner. If any row for a
        person has a value of 4 in the "selection_partners" column, all rows for that person are inferred to be in a
        relationship. Otherwise, the relationship status is set to 0.

        Args:
            df_states (pd.DataFrame): The DataFrame containing the ESM data with interaction information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'relationship' indicating inferred relationship status.
        """
        # Parse the configuration for the relationship status (can be useful for future expansion)
        relationship_cfg = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                              "binary",
                                              "relationship")[0]
        ia_partner_col = relationship_cfg["item_names"]["cocoesm"]
        ia_partner_val = relationship_cfg["special_mappings"]["cocoesm"]

        # Check if 'selection_partners' exists in the DataFrame
        if ia_partner_col in df_states.columns:
            # Apply the _map_comma_separated function to each row to check for interaction
            df_states['interaction'] = df_states[ia_partner_col].apply(
                lambda x: self._map_comma_separated(x, {ia_partner_val: 1})
            )

            # Group by person ID (self.raw_esm_id_col) and check if any interaction occurred for the person
            partner_interaction = df_states.groupby(self.raw_esm_id_col)['interaction'].transform('max')

            # Assign 1 to 'relationship' if the person interacted with their partner in any row, otherwise 0
            df_states['relationship'] = np.where(partner_interaction == 1, 1, 0)

            # Drop the intermediate 'interaction' column
            df_states.drop(columns=['interaction'], inplace=True)
        else:
            raise KeyError(f"Column {ia_partner_col} not in {self.dataset}")

        return df_states

    def merge_trait_df_country_vars(self, country_var_dct: dict[str, pd.DataFrame], df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method
            a) merges the 3 country-level var dfs (health, psycho-political, socio-economic)
            b) merges the df_traits with the country-level df
        For the second step, we need to create a column that contains the year of the assessment. Because the assessment
        is about several weeks, it is possible that some participants started in one year and ended in a second year (but this
        case is negligible small, only around 50 participants are affected). Therefore, we just take the "demographics_created_t1"
        column as a reference for the year of participation.


        Args:
            country_var_dct:
            df_traits:

        Returns:

        """
        df_country_level = reduce(
            lambda left, right: pd.merge(
                left=left,
                right=right,
                on=["country", "year"],
                how="outer"),
            country_var_dct.values())
        df_country_level.columns = [f"{col}" for col in df_country_level.columns]  # add column prefix
        df_traits["year"] = pd.to_datetime(df_traits["created_demographics"]).dt.year
        df_traits = pd.merge(left=df_traits,
                             right=df_country_level,
                             on=["country", "year"],
                             how="left")  # TODO which join?
        return df_traits
