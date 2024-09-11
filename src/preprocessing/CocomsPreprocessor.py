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
        close_interaction_cfg = [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                                 if entry["name"] == "close_interactions"][0]  # TODO write function that does return the cfg part
        wave1_weak_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"]["wave1"]["weak_ties"]
        wave1_close_ties_col_lst = close_interaction_cfg["special_mappings"]["cocoms"]["wave1"]["close_ties"]
        wave3_mapping = close_interaction_cfg["special_mappings"]["cocoms"]["wave3"]

        # Handle Wave 2
        df_states.loc[df_states['studyWave'] == 2, 'close_interactions_raw'] = np.nan
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

        df_states['close_interactions'] = df_states['unique_id'].map(interaction_stats)
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
        # TODO: Modularize further if time
        # Define the conversation topic configurations
        conversation_configs = {
            "work_conversations": {
                "cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                         if entry["name"] == "work_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["work_columns"],
                "other_cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                               if entry["name"] == "work_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
            },
            "personal_conversations": {
                "cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                         if entry["name"] == "personal_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["pers_columns"],
                "other_cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                               if entry["name"] == "personal_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
            },
            "societal_conversations": {
                "cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                         if entry["name"] == "societal_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["soc_columns"],
                "other_cols": [entry for entry in self.fix_cfg["predictors"]["self_reported_micro_context"]
                               if entry["name"] == "societal_conversations"][0]["special_mappings"]["cocoms"]["wave1"]["other_columns"]
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

        df_states.loc[df_states['studyWave'] != 1, ['work_conversations', 'personal_conversations', "societal_conversations"]] = np.nan

        # Calculate percentage of each conversation type per unique_id for CoCo MS1
        for col in ['work_conversations', 'personal_conversations', 'societal_conversations']:
            interaction_stats = df_states.groupby('unique_id')[col].apply(
                lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
            )
            df_states[col] = df_states['unique_id'].map(interaction_stats)

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
        relationship_cfg = self.config_parser(self.fix_cfg["predictors"]["self_reported_micro_context"],
                                              "binary", "relationship")

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
                # Create a boolean mask for partner interaction in the respective study wave
                partner_interaction = df_states.groupby(self.raw_esm_id_col)[ia_partner_col].transform(
                    lambda x: (x == ia_partner_val).any())

                # Apply the result to the 'relationship' column for the respective study wave
                df_states.loc[df_states['StudyWave'] == wave, 'relationship'] = np.where(partner_interaction, 1, 0)
            else:
                raise KeyError(f"Column {ia_partner_col} not in {self.dataset}")

        return df_states



