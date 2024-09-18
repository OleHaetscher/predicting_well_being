import re
from copy import deepcopy
from datetime import timedelta

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
        self.relationship = None  # will be assigned and holded
        self.work_conversations = None
        self.personal_conversations = None
        self.societal_conversations = None

        # Track persons that are in multiple waves
        self.person_in_two_waves = None
        self.person_in_three_waves = None

    def merge_traits(self, df_dct):
        df_dct_proc = self.cocoms_processing_before_merging(deepcopy(df_dct))
        traits_dfs = [df for key, df in df_dct_proc.items() if 'traits' in key]
        concatenated_traits = pd.concat(traits_dfs, axis=0).reset_index(drop=True)
        return concatenated_traits

    def cocoms_processing_before_merging(self, df_dct: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        This is some rather dumb manual processing of the cocoms dfs before I can reliably merge them to one df

        Args:

        Returns:

        """
        df_w1 = df_dct["data_traits_w1"]
        df_w2 = df_dct["data_traits_w2"]
        df_w3 = df_dct["data_traits_w3"]

        # Rename hexaco cols (hex_60 in w1,w2; hex60 in w3)
        df_w3.columns = [re.sub(r'^hex60', 'hex_60', col) for col in df_w3.columns]
        test = [i for i in df_w3.columns if "professional" in i]

        # Align professional_status col and align categorical values
        df_w1["professional_status_t1"] = df_w1["professional_status_student_t1"].apply(lambda x: 5 if x in [2, 3, 4] else 0)  # part-time
        df_w2["professional_status_t1"] = df_w2["professional_status_student_t1"].apply(lambda x: 5 if x in [2, 3, 4] else 0)  # part-time
        df_w3["professional_status_t1"] = df_w3["professional_status_booster_t1"]
        df_w1["professional_status_student_t1"] = 1
        df_w2["professional_status_student_t1"] = 1
        df_w3["professional_status_student_t1"] = pd.notna(df_w3["professional_status_student_t1"]).astype(int)  # 1 / 0

        df_dct["data_esm_w1"] = df_w1
        df_dct["data_esm_w2"] = df_w2
        df_dct["data_esm_w3"] = df_w3
        return df_dct

    def merge_states(self, df_dct):
        esm_dfs = [df for key, df in df_dct.items() if 'esm' in key]
        concatenated_esm = pd.concat(esm_dfs, axis=0).reset_index(drop=True)
        return concatenated_esm

    def clean_trait_col_duplicates(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Removes a specified suffix from all column names in the DataFrame if the suffix is present.
        Additionally, when columns with "_t1" and "_t2" suffixes conflict, the function fills missing
        values in "_t1" with values from "_t2" and then drops the "_t2" column.

        Args:
            df_traits: A pandas DataFrame whose column names need to be updated.

        Returns:
            A pandas DataFrame with the updated column names and values.
        """
        trait_suffixes = ["_t1", "_t2"]
        updated_columns = []
        columns_to_fill = {}  # Dictionary to track columns that need filling

        # Iterate over columns
        for col in df_traits.columns:
            # Track the original column name before stripping suffix
            original_col = col

            # Iterate over suffixes and remove them if applicable
            for suffix in trait_suffixes:
                if col.endswith(suffix):
                    base_col_name = col[:-len(suffix)]

                    # If the column is "_t2" and base name already exists in updated_columns
                    if suffix == "_t2" and base_col_name in updated_columns:
                        # Add to the list of columns to fill from _t2 to _t1
                        columns_to_fill[base_col_name] = original_col
                        break  # Skip adding "_t2" column as it will be merged into "_t1"

                    col = base_col_name

            updated_columns.append(col)

        # Fill NaN values in _t1 columns using _t2 columns
        for base_col, t2_col in columns_to_fill.items():
            t1_col = f"{base_col}_t1"
            if t1_col in df_traits.columns:
                # Fill NaN values in the _t1 column with values from the corresponding _t2 column
                df_traits[t1_col].fillna(df_traits[t2_col], inplace=True)
                # Drop the _t2 column after merging
                # df_traits.drop(columns=[t2_col], inplace=True)

        df_traits.columns = updated_columns

        # Check for duplicates after renaming
        assert len(df_traits.columns) == len(set(df_traits.columns)), "Duplicate column names found after renaming!"

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
        party_number_map = [entry["party_num_mapping"] for entry in self.fix_cfg["person_level"]["personality"]
                            if "party_num_mapping" in entry.keys()][0]["cocoms"]
        df_traits['political_orientation'] = df_traits['vote_general'].map(party_number_map).fillna(np.nan)
        return df_traits

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases.

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        df_states = self.create_close_interactions(df_states=df_states)
        df_states = self.create_conversation_topic_columns(df_states=df_states)
        df_states = self.create_relationship(df_states=df_states)
        df_states = self.adjust_number_interactions_col(df_states=df_states)
        return df_states

    def create_close_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In CoCoMS, there are differences in how the type of interaction partner is assessed between study waves:
            - In CoCo MS1, it is assessed with multiple binary variables (e.g., "selection_partners_05").
            - In CoCo MS2, it is not assessed at all (selection_partners_01 contains only the continuous assignment).
            - In CoCo MS3, it is assessed with one categorical ("selection_partners_01").

        This method creates two columns:
            - "close_interactions_raw": where 1 == close ties, 0 == weak ties, and NaN for conflicts or unassessed interactions.
            - "close_interactions": a percentage ratio of close ties (1) to total interactions (1 and 0) per person as denoted by "unique_id".

        Processing steps:
            - In CoCo MS1, the binary columns for interaction partners are mapped if the partners are unambiguously close or weak ties.
            - In CoCo MS2, "close_interactions_raw" is set to NaN because no interaction types are assessed.
            - In CoCo MS3, categorical values are mapped to binary in "close_interactions_raw", and -1 (no interaction) is set to NaN
              (because -1 stands for "no_interaction" and thus must not influence the percentage)
            - Finally, "close_interactions" is calculated as the ratio of close ties to total ties (close + weak) for each person.

        Args:
            df_states (pd.DataFrame): The DataFrame containing the interaction data for all study waves.

        Returns:
            pd.DataFrame: The modified DataFrame with "close_interactions_raw" and "close_interactions" columns.
        """
        close_interaction_cfg = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                                   "percentage",
                                                   "close_interactions"
                                                   )[0]
        wave1_weak_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"]["wave1"]["weak_ties"]
        wave1_close_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"]["wave1"]["close_ties"]
        wave3_mapping = close_interaction_cfg["special_mappings"]["cocoms"]["wave3"]


        # Handle Wave 1
        close_tie_mask = df_states.loc[df_states['studyWave'] == 1, wave1_close_ties_col_lst].max(axis=1) == 1
        weak_tie_mask = df_states.loc[df_states['studyWave'] == 1, wave1_weak_ties_col_lst].max(axis=1) == 1
        df_states.loc[df_states['studyWave'] == 1, 'close_interactions_raw'] = np.where(
            close_tie_mask & ~weak_tie_mask, 1,
            np.where(weak_tie_mask & ~close_tie_mask, 0, np.nan)
        )

        # Handle Wave 3
        df_states.loc[df_states['studyWave'] == 3, 'close_interactions_raw'] = df_states.loc[
            df_states['studyWave'] == 3, 'selection_partners_01'].replace(wave3_mapping)
        df_states.loc[(df_states['studyWave'] == 3) & (df_states['close_interactions_raw'] == -1),
        'close_interactions_raw'] = np.nan

        # Create percentage
        interaction_stats = df_states.groupby('unique_id')['close_interactions_raw'].apply(
            lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
        )

        # Handle Wave 2
        df_states.loc[df_states['studyWave'] == 2, 'close_interactions'] = np.nan

        df_states['close_interactions'] = df_states['unique_id'].map(interaction_stats)

        self.close_interactions = deepcopy(df_states[['close_interactions', self.raw_esm_id_col]].drop_duplicates(keep="first"))
        return df_states

    def create_conversation_topic_columns(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates the columns
            - work_conversations
            - personal_conversations
            - societal_conversations
        There are again differences between the cocoms waves in the assessment of the variable "selection_topic"
            - In CoCo MS1, all categories are assessed
            - In CoCo MS2, only societal conversations are assessed
            - In CoCo MS3, only societal conversations are assessed
        Thus, we can only calculate the percentages for CoCo MS1. The values for CoCo MS2 and MS3 are set to np.nan

        Args:
            df_states:

        Returns:

        """
        conv_topic_vars = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                                     "percentage",
                                                     "work_conversations",
                                                     "personal_conversations",
                                                     "societal_conversations"
                                                     )
        # Define the conversation topic configurations
        conversation_configs = {
            "work_conversations": {
                "cols": conv_topic_vars[0]["special_mappings"]["cocoms"]["wave1"]["work_columns"],
                "other_cols": conv_topic_vars[0]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
            },
            "personal_conversations": {
                "cols": conv_topic_vars[1]["special_mappings"]["cocoms"]["wave1"]["pers_columns"],
                "other_cols": conv_topic_vars[1]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
            },
            "societal_conversations": {
                "cols": conv_topic_vars[2]["special_mappings"]["cocoms"]["wave1"]["soc_columns"],
                "other_cols": conv_topic_vars[2]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
            }
        }

        # Loop through the conversation types and create the respective columns
        for conversation_type, config in conversation_configs.items():
            topic_mask = (
                    (df_states[config["cols"]].max(axis=1) == 1) & (df_states[config["other_cols"]].max(axis=1) == 0)
            )
            other_mask = (
                    (df_states[config["other_cols"]].max(axis=1) == 1) & (df_states[config["cols"]].max(axis=1) == 0)
            )
            df_states[conversation_type] = np.where(topic_mask, 1, np.where(other_mask, 0, np.nan))
            cols = [i for i in df_states.columns if "selection_topics" in i] + [conversation_type, "unique_id"]

        # the topics were only assessed in wave1
        df_states.loc[df_states['studyWave'] != 1, ['work_conversations', 'personal_conversations',
                                                    "societal_conversations"]] = np.nan

        # Calculate percentage of each conversation type per unique_id for CoCo MS1
        for col in ['work_conversations', 'personal_conversations', 'societal_conversations']:
            interaction_stats = df_states.groupby('unique_id')[col].apply(
                lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
            )
            df_states[col] = df_states['unique_id'].map(interaction_stats)

        self.work_conversations = deepcopy(df_states[["work_conversations", self.raw_esm_id_col]].drop_duplicates(keep="first"))
        self.personal_conversations = deepcopy(df_states[["personal_conversations", self.raw_esm_id_col]].drop_duplicates(keep="first"))
        self.societal_conversations = deepcopy(df_states[["societal_conversations", self.raw_esm_id_col]].drop_duplicates(keep="first"))
        return df_states

    def create_relationship(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Infers the relationship status from the ESM surveys based on interactions with a partner. If a person interacted
        with their partner in StudyWave 1 or 3, all rows for that person will indicate a relationship. StudyWave 2 will
        be set to np.nan.

        Args:
            df_states (pd.DataFrame): The DataFrame containing the ESM data with interaction information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'relationship' indicating inferred relationship status.
        """

        # Parse the configuration for the relationship status
        relationship_cfg = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                              "binary", "relationship")[0]

        # Mapping the study waves, column names, and partner interaction values for waves 1 and 3
        waves_config = {
            1: {
                'column': relationship_cfg["item_names"][self.dataset]["wave1"],
                'partner_value': relationship_cfg["special_mappings"][self.dataset]["wave1"]
            },
            3: {
                'column': relationship_cfg["item_names"][self.dataset]["wave3"],
                'partner_value': relationship_cfg["special_mappings"][self.dataset]["wave3"]
            }
        }

        # Initialize the 'relationship' column with np.nan by default (this will hold for StudyWave 2)
        df_states['relationship'] = np.nan

        # Iterate through StudyWaves 1 and 3 using the configuration
        for wave, config in waves_config.items():
            ia_partner_col = config['column']
            ia_partner_val = config['partner_value']

            # Check if the column exists
            if ia_partner_col in df_states.columns:
                # Check if a person interacted with their partner at least once
                partner_interaction = (df_states
                                       .groupby(self.raw_esm_id_col)[ia_partner_col]
                                       .agg(lambda x: (x == ia_partner_val).any()
                                            )
                                       )
                # Assign the relationship status for the respective study wave
                df_states.loc[df_states['studyWave'] == wave, 'relationship'] = df_states[self.raw_esm_id_col].map(
                    partner_interaction).astype(int)
            else:
                raise KeyError(f"Column {ia_partner_col} not in {self.dataset}")

        self.relationship = deepcopy(df_states[["relationship", self.raw_esm_id_col]].drop_duplicates(keep="first"))
        return df_states

    def adjust_number_interactions_col(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the 'selection_medium_00' column based on the time difference between
        'time_social_interaction' and the questionnaire time ('self.esm_timestamp').
        If the interaction occurred more than one hour before the questionnaire, the
        corresponding value in 'selection_medium_00' will be set to 0.

        The 'time_social_interaction' column is in fractional hours format (e.g., 16.75 = 16:45),
        and the 'questionnaireStartedTimestamp' is the time the survey was started.

        Args:
            df_states (pd.DataFrame): DataFrame containing the following columns:
                - 'questionnaireStartedTimestamp': The timestamp when the questionnaire was started.
                - 'time_social_interaction': Time of the interaction, represented in fractional hours.
                - 'selection_medium_00': Column indicating if an interaction took place (1 for yes, 0 for no).

        Returns:
            pd.DataFrame: The adjusted DataFrame with updated 'selection_medium_00' column.
        """

        # Ensure 'questionnaireStartedTimestamp' is in datetime format and fix format differences (remove milliseconds)
        df_states[self.esm_timestamp] = df_states[self.esm_timestamp].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
        df_states[f"{self.esm_timestamp}_dt"] = pd.to_datetime(df_states[self.esm_timestamp])

        # Convert 'questionnaireStartedTimestamp' to Timedelta (hours, minutes, seconds of the day)
        df_states['esm_timedelta'] = df_states[f"{self.esm_timestamp}_dt"].dt.hour * 3600 + \
                                     df_states[f"{self.esm_timestamp}_dt"].dt.minute * 60 + \
                                     df_states[f"{self.esm_timestamp}_dt"].dt.second
        df_states['esm_timedelta'] = pd.to_timedelta(df_states['esm_timedelta'], unit='s')

        # Convert 'time_social_interaction' from fractional hours to a timedelta object
        df_states['interaction_time'] = df_states['time_social_interaction'].apply(
            lambda x: timedelta(hours=int(x), minutes=int((x % 1) * 60)) if pd.notna(x) else pd.NaT
        )

        # Adjust 'selection_medium_00' based on whether the interaction happened within 1 hour of the questionnaire
        df_states['selection_medium_00'] = df_states.apply(
            lambda row: 1 if (row['esm_timedelta'] - row['interaction_time']) > timedelta(hours=1)
            else row['selection_medium_00'],
            axis=1
        )
        return df_states

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        No custom adjustments necessary in cocoesm.

        Args:
            df:

        Returns:
            pd.DataFrame
        """
        df = df.merge(self.relationship, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.close_interactions, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.work_conversations, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.personal_conversations, on=self.raw_esm_id_col, how="left")
        df = df.merge(self.societal_conversations, on=self.raw_esm_id_col, how="left")
        return df
