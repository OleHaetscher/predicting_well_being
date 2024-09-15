import os

import pandas as pd


class PrelimPreprocessor:
    """
    This class applies some custom preprocessing to align the different ESM datasets.
    This includes assigning a common id column, reducing column size, and aligning the format.
    At the end of its method application, its attribute "data_dct" will contain the ESM datasets in
    the following format:
    {cocoms: {traits: pd.DataFrame, esm: pd.DataFrame}, cocoesm: ....}
    """

    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Initializes the PrelimPreprocessorwith a configuration file.

        Args:
            cfg (str): yaml config
        """
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.datasets = {}

    @property
    def dataset_names(self):
        return self.var_cfg["prelimpreprocessor"]["datasets"]

    @property
    def path_to_raw_data(self):
        return self.var_cfg["prelimpreprocessor"]["path_to_raw_data"]

    def prelimpreprocessing(self):
        for dataset_name in self.dataset_names:
            path_to_dataset = os.path.join(self.path_to_raw_data, dataset_name)
            df_dct = self.load_data(path_to_dataset=path_to_dataset)
            df_dct = self.set_common_id_as_index(df_dct=df_dct, dataset_name=dataset_name)
            df_dct_processed = self.custom_dataset_processing(df_dct=df_dct, dataset_name=dataset_name)
            self.datasets[dataset_name] = df_dct_processed
        print()

    def load_data(self, path_to_dataset):
        files = os.listdir(path_to_dataset)
        if files:
            df_dct = {file[:-4]: pd.read_csv(os.path.join(path_to_dataset, file), encoding="latin", nrows=1000)
                      for file in files}
            return df_dct
        else:
            raise FileNotFoundError(f"Not datasets found in {self.path_to_raw_data}")

    def set_common_id_as_index(self, df_dct: dict[str, pd.DataFrame], dataset_name: str) -> dict[str, pd.DataFrame]:
        """
        Set a common ID column as the index for dataframes in df_dct that contain "traits" in the key
        and do not contain "esm".

        Args:
            df_dct (dict): Dictionary containing the dataframes.
            dataset_name (str): Name of the dataset to map the raw ID column.

        Returns:
            dict: Updated dictionary with the common ID column set as index for relevant dataframes.
        """
        raw_id_col = self.var_cfg["prelimpreprocessor"]['raw_id_cols'].get(dataset_name)
        common_id_col = self.var_cfg["prelimpreprocessor"]['common_id_col']

        if not raw_id_col:
            raise ValueError(f"No raw ID column specified for dataset {dataset_name}")

        updated_df_dct = {}
        for key, df in df_dct.items():
            if "esm" not in key:  # currently, this excludes the emotions dataset
                try:
                    df[common_id_col] = df[raw_id_col]
                except KeyError:
                    print(f"Key Error, no ID column was set as index for {key}, {common_id_col} already exists")
                df.set_index(common_id_col, inplace=True)
            updated_df_dct[key] = df

        return updated_df_dct


    def custom_dataset_processing(self, df_dct, dataset_name):
        """This is messy, maybe adjust later."""
        if dataset_name in ['cocoesm', "pia"]:  # already traits (data_traits) and esm (data_esm)
            return df_dct
        elif dataset_name == "cocoms":
            return self.process_cocoms(df_dct)
        elif dataset_name == "cocout":
            return self.process_cocout(df_dct)
        elif dataset_name == "emotions":
            return self.process_emotions(df_dct)
        elif dataset_name == "zpid":
            return self.process_zpid(df_dct)
        else:
            raise NameError(f"Unknown dataset: {dataset_name}")


    @staticmethod
    def process_cocoms(df_dct: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Custom processing for cocoms: concatenate the data from different waves along the rows.

        Args:
            df_dct (dict): Dictionary containing the raw dataframes from cocoms.

        Returns:
            dict: Dictionary containing "traits" and "esm" as keys and the associated pd.DataFrames as values.
        """
        esm_dfs = [df for key, df in df_dct.items() if 'esm' in key]
        traits_dfs = [df for key, df in df_dct.items() if 'traits' in key]

        concatenated_esm = pd.concat(esm_dfs, axis=0).reset_index(drop=True)
        concatenated_traits = pd.concat(traits_dfs, axis=0).reset_index(drop=True)

        return {
            "data_esm": concatenated_esm,
            "data_traits": concatenated_traits
        }

    @staticmethod
    def process_cocout(df_dct: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Custom processing for cocout dataset: concatenate data along rows and columns based on the dataset type.

        Args:
            df_dct (dict): Dictionary containing the raw dataframes from cocout dataset.

        Returns:
            dict: Dictionary containing concatenated dataframes for "data_esm" and "data_traits".
        """
        # Concatenate data_esm along rows
        esm_dfs = [df for key, df in df_dct.items() if 'data_esm' in key]
        concatenated_esm = pd.concat(esm_dfs, axis=0).reset_index(drop=True)

        coco_ut1_traits = (
            df_dct["data_traits_t1_ut1"].merge(
                df_dct["data_traits_t2_ut1"], left_index=True, right_index=True, how="outer", suffixes=(None, "_postsurv"))
            .merge(df_dct["data_personality_ut1"], left_index=True, right_index=True, how="outer", suffixes=(None, "_pers"))
            .merge(df_dct["data_demographics_ut1"], left_index=True, right_index=True, how="outer", suffixes=(None, "_dem"))
                   )

        coco_ut2_traits = (
            df_dct["data_traits_t1_ut2"].merge(
                df_dct["data_traits_t2_ut2"], left_index=True, right_index=True, how="outer", suffixes=(None, "_postsurv"))
            .merge(df_dct["data_personality_ut2"], left_index=True, right_index=True, how="outer", suffixes=(None, "_pers"))
            .merge(df_dct["data_demographics_ut1"], left_index=True, right_index=True, how="outer",
                   suffixes=(None, "_dem"))
        )
        df_coco_ut_traits = pd.concat([coco_ut1_traits, coco_ut2_traits], axis=0)

        return {
            "data_esm": concatenated_esm,
            "data_traits": df_coco_ut_traits
        }

    @staticmethod
    def process_emotions(df_dct: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Process the emotions data: split the dataframe into two - one with columns that vary by ID
        and one with columns that do not vary by ID (which corresponds to person-level and esm variables)

        Args:
            df_dct (dict): Dictionary containing the raw dataframes.

        Returns:
            dict: A dictionary with two dataframes: 'varying_df' and 'non_varying_df'.
        """
        df_raw = df_dct["data_traits_esm"]
        id_column = "id_for_merging"

        # Determine columns that do not vary by ID
        trait_columns = [
            col for col in df_raw.columns if col != id_column and df_raw.groupby(id_column)[col].nunique().max() == 1
        ]
        esm_columns = [col for col in df_raw.columns if col not in trait_columns and col != id_column]
        trait_df = df_raw[[id_column] + trait_columns].drop_duplicates().reset_index(drop=True)
        esm_df = df_raw[[id_column] + esm_columns].reset_index(drop=True)

        return {
            "data_esm": esm_df,
            "data_traits": trait_df
        }

    @staticmethod
    def process_zpid(df_dct):
        pass








