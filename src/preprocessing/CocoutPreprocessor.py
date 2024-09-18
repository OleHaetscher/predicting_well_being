from src.preprocessing.BasePreprocessor import BasePreprocessor
import pandas as pd

class CocoutPreprocessor(BasePreprocessor):
    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Constructor method of the LassoAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(fix_cfg, var_cfg)
        self.dataset = "cocout"

    def merge_traits(self, df_dct):
        dataset_lst = []
        for dataset in ["ut1", "ut2"]:
            # Specify the unspecific and overlapping col names for the different UT surveys and clear ID columns
            df_dct[f"data_personality_{dataset}"] = df_dct[f"data_personality_{dataset}"].add_suffix('_bfi')
            df_dct[f"data_personality_{dataset}"] = df_dct[f"data_personality_{dataset}"].dropna(subset="pID_bfi")
            df_dct[f"data_self_esteem_{dataset}"] = df_dct[f"data_self_esteem_{dataset}"].add_suffix('_se')
            df_dct[f"data_self_esteem_{dataset}"] = df_dct[f"data_self_esteem_{dataset}"].dropna(subset="pID_se")
            df_dct[f"data_demographics_{dataset}"] = df_dct[f"data_demographics_{dataset}"].add_suffix('_dem')
            df_dct[f"data_demographics_{dataset}"] = df_dct[f"data_demographics_{dataset}"].dropna(subset="pID_dem")

            # some variable names differ across waves -> Fix this here
            self.fix_col_name_issues(df_dct, dataset)

            # Merge Dataframes along the columns
            df_traits = (
                df_dct[f"data_traits_t1_{dataset}"]
                .merge(df_dct[f"data_traits_t2_{dataset}"], left_on="id", right_on="id", how="outer", validate='1:1',
                       suffixes=(None, "_postsurv"))
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(df_dct[f"data_personality_{dataset}"], left_on="id", right_on="pID_bfi", how="outer", validate='1:1')
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(df_dct[f"data_demographics_{dataset}"], left_on="id", right_on="pID_dem", how="outer", validate='1:m')
                .dropna(subset="id")
                .reset_index(drop=True)
                .merge(df_dct[f"data_self_esteem_{dataset}"], left_on="id", right_on="pID_se", how="outer", validate='1:1')
                .dropna(subset="id")
                .reset_index(drop=True)
            )
            dataset_lst.append(df_traits)
        # Concat Dataframes along the rows (ut1 and ut2)
        concatenated_traits = pd.concat(dataset_lst, axis=0)
        return concatenated_traits

    def fix_col_name_issues(self, df_dct: dict, dataset):
        """
        There are some colname differences between CoCo UT1 and CoCo UT2. We fix this before merging the data

        Args:
            df_dct:
            dataset: "ut1" or "ut2

        Returns:
            dict:
        """
        if dataset == "ut2":
            df_t1 = df_dct[f"data_traits_t1_{dataset}"]
            df_t2 = df_dct[f"data_traits_t2_{dataset}"]
            cmq_cols = [col for col in df_t1.columns if "cms_" in col]
            df_t1 = df_t1.rename(columns={col: col.replace("cms_", "cmq_") for col in cmq_cols})
            df_t2 = df_t2.rename(columns={col: col.replace("cms_", "cmq_") for col in cmq_cols})
            df_dct[f"data_traits_t1_{dataset}"] = df_t1
            df_dct[f"data_traits_t2_{dataset}"] = df_t2

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
        In CoCo UT, we need to create the columns for professional_status and educational_attainment. We just
        assign an array of 1s as column values. The assignment to the binary variables is handled in the config.

        Args:
            df_traits:

        Returns:
            pd.DataFrame:
        """
        df_traits["professional_status"] = 1
        df_traits["educational_attainment"] = 6  # higher secondary education in 1-10 scale
        return df_traits

    def merge_states(self, df_dct):
        df_lst = []
        for dataset in ["ut1", "ut2"]:
            data_esm = df_dct[f"data_esm_{dataset}"]
            data_esm_daily = df_dct[f"data_esm_daily_{dataset}"]
            data_esm['date'] = pd.to_datetime(data_esm['RecordedDateConvert']).dt.date
            data_esm_daily['date'] = pd.to_datetime(data_esm_daily['RecordedDateConvert']).dt.date
            # Merge the DataFrames based on id and date column
            merged_df = pd.merge(
                data_esm,
                data_esm_daily,
                how='left',
                on=['id', 'date'],
                suffixes=("_esm", "_daily"),
            )
            df_lst.append(merged_df)
        concatenated_esm = pd.concat(df_lst, axis=0).reset_index(drop=True)
        return concatenated_esm

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases.

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        df_states = self.clean_number_interaction_partners(df_states=df_states)
        # df_states = self.clean_days_infected(df_states=df_states)
        return df_states

    def clean_number_interaction_partners(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In CoCo UT, we need to
            - convert unambiguous strings (e.g., "one) to numerics
            - set ambiguous strings in the "number_interaction_partners" columns to np.nan

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        number_int_partners_cfg = [entry for entry in self.fix_cfg["esm_based"]["self_reported_micro_context"]
                                   if entry["name"] == "number_interaction_partners"][0]
        col_name = number_int_partners_cfg["item_names"]["cocout"]
        df_states[col_name] = df_states[col_name].replace(
            number_int_partners_cfg["special_mappings"]["cocout"]["str_to_num"])
        df_states[col_name] = pd.to_numeric(df_states[col_name], errors='coerce')
        return df_states

