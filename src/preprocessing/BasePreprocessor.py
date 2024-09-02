import copy
import os
from abc import ABC, abstractmethod
from typing import Union, Callable, Any

import numpy as np
import pandas as pd
from src.utils.logger import Logger
from src.utils.timer import Timer

class BasePreprocessor(ABC):
    """
    """

    def __init__(self, fix_cfg: dict, var_cfg: dict):
        """
        Initializes the BasePreprocessor with a configuration file.

        Args:
            fix_cfg (dict): yaml config
            var_cfg (dict): yaml config
        """
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.dataset = None  # assigned in the subclasses
        self.logger = Logger(log_dir=self.var_cfg["general"]["log_dir"], log_file=self.var_cfg["general"]["log_name"])
        self.timer = Timer(self.logger)  # composition
        self.data = None  # assigned at the end, preprocessed data
        self.apply_preprocessing_methods = self.timer._decorator(self.apply_preprocessing_methods)

    @property
    def path_to_raw_data(self):
        """Path to the folder containing the raw files for self.dataset."""
        return os.path.join(self.var_cfg["prelimpreprocessor"]["path_to_raw_data"], self.dataset)

    def apply_preprocessing_methods(self):
        """

        Returns:

        """
        self.logger.log(f".")
        self.logger.log(f"Starting preprocessing pipeline for >>>{self.dataset}<<<")

        # Step 1: Load data
        df_dct = self.load_data(path_to_dataset=self.path_to_raw_data)
        df_traits = None  # Initialize df_traits as None
        df_states = None

        # Step 2: Process and transform trait data
        preprocess_steps_traits = [
            (self.merge_traits, {'df_dct': df_dct}),
            (self.clean_trait_col_duplicates, {'df_traits': None}),
            (self.exclude_flagged_rows, {'df_traits': None}),
            (self.dataset_specific_trait_processing, {'df_traits': None}),
            (self.select_trait_vars, {'df_traits': None}),
            # (self.dataset_specific_trait_processing, {'df_traits': None}),
            (self.align_scales, {
                'df': None,
                'df_type': "traits",
                'var_type': "predictor",
                'cat_list': ['personality', 'sociodemographics'],
            }),
            (self.create_binary_vars_from_categoricals, {'df_traits': None}),
        ]
        for method, kwargs in preprocess_steps_traits:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_traits for k, v in kwargs.items()}
            df_traits = self._log_and_execute(method, **kwargs)

        # Step 3: Process and transform esm data
        preprocess_steps_esm = [
            (self.merge_states, {'df_dct': df_dct}),
            (self.dataset_specific_state_processing, {'df_traits': None}),
            (self.select_trait_vars, {'df_traits': None}),
            # (self.dataset_specific_trait_processing, {'df_traits': None}),
            (self.align_scales, {
                'df': None,
                'df_type': "traits",
                'var_type': "predictor",
                'cat_list': ['personality', 'sociodemographics'],
            }),
            (self.create_binary_vars_from_categoricals, {'df_traits': None}),
        ]


        self.logger.log(f"Finished preprocessing pipeline for >>>{self.dataset}<<<")
        return df_traits

    def _log_and_execute(self, method: Callable, *args: Any, **kwargs: Any):
        self.logger.log(f"   Executing {method.__name__}")
        return method(*args, **kwargs)

        print()
        #df_traits = self.dataset_specific_processing_traits(df_traits)
        #df_traits = self.inverse_code_items(df_traits)
        #df_traits = self.compute_scale_means(df_traits)

        #df_esm = self.merge_states(df_dct=df_dct)
        #df_esm = self.select_state_vars(df_esm)
        #df_esm = self.dataset_specific_esm_processing(df_esm)
        #df_states = self.dataset_specific_processing_states(df_states)
        # inverse coding / scale mean calculation necessary?
        #df_states = self.create_pl_state_vars(df_states)



    def load_data(self, path_to_dataset):
        """
        This method loads all files contained in self.path_to_raw_data and returns a dict containing pd.DataFrames

        Args:
            path_to_dataset:

        Returns:


        """
        files = os.listdir(path_to_dataset)
        if files:
            df_dct = {file[:-4]: pd.read_csv(os.path.join(path_to_dataset, file), encoding="latin", nrows=5000)
                      for file in files}
            return df_dct
        else:
            raise FileNotFoundError(f"Not datasets found in {self.path_to_raw_data}")

    @abstractmethod
    def merge_traits(self, df_dct):
        """
        This method merges the files found in self.path_to_raw_data so that the resulting dataframe contains
        all trait data for self.dataset. This must be implemented in the subclasses.

        Args:
            df_dct (dict): Dict containing the filenames as keys and the pd.DataFrames as values

        Returns:
            pd.DataFrame: df containing all trait variables for a certain dataset
        """
        pass

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
    pass

    def exclude_flagged_rows(self, df_traits) -> pd.DataFrame:
        """
        This function excludes persons from the trait df that indicated that they did
        responded careless.

        Args:
            df_traits:

        Returns:

        """
        flags = [entry for entry in self.fix_cfg["other_columns"] if entry["name"] == "trait_flags"][0]
        flag_col = flags["item_names"][self.dataset]
        df_traits[flag_col] = df_traits[flag_col].map(flags["category_mappings"][self.dataset]).fillna(np.nan)
        flag_filtered_df = df_traits[df_traits[flag_col] == 1]
        return flag_filtered_df

    def select_trait_vars(self, df_traits) -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns relevant to the specified dataset.

        Args:
            df_traits: A pandas DataFrame containing the dataset.

        Returns:
            pd.DataFrame: A DataFrame filtered to include only the relevant columns.
        """
        columns_to_be_selected = []
        # predictors
        for category, traits in self.fix_cfg["predictors"]["person_level"].items():
            for trait in traits:
                columns_to_be_selected.extend(self.extract_columns(trait['item_names']))
        # criterion - trait wb items
        if self.dataset in self.fix_cfg["criterion"]["traits"]:
            trait_pa_columns = self.fix_cfg["criterion"]["traits"][self.dataset]["pa"]
            trait_na_columns = self.fix_cfg["criterion"]["traits"][self.dataset]["na"]
            columns_to_be_selected.extend(trait_pa_columns + trait_na_columns)
        df_col_filtered = df_traits[list(set(columns_to_be_selected))]
        return df_col_filtered

    def dataset_specific_trait_processing(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases.

        Args:
            df_traits:

        Returns:
            pd.DataFrame:
        """
        return df_traits

    def align_scales(self, df: pd.DataFrame, cat_list = list, df_type: str = 'traits', var_type: str = 'predictor') -> pd.DataFrame:
        """
        Aligns the scales of specified columns in the DataFrame according to the configuration.

        This method searches through the configuration and processes only those entries where
        `self.dataset` has a corresponding key in `align_scales_mapping`. For each such entry,
        it aligns the numerical values of the columns listed in `item_names[self.dataset]` based
        on the scaling mapping provided in `align_scales_mapping[self.dataset]`.

        Args:
            df: The DataFrame containing either trait or esm data.
            df_type: "traits" or "esm"
            feature_categories: variable number of feature category strings

        Returns:
            pd.DataFrame: The DataFrame with the aligned scales.
        """
        if var_type == 'predictor':
            specific_cfg = self.fix_cfg["predictors"]
            if df_type == 'traits':
                specific_cfg = specific_cfg["person_level"]
            elif df_type == 'esm':
                pass
            else:
                raise ValueError("df_type must be either 'traits' or 'esm'")

            for cat in cat_list:
                for entry in specific_cfg[cat]:
                    item_names = entry['item_names'].get(self.dataset)
                    align_mapping = entry.get('align_scales_mapping', {}).get(self.dataset)

                    if item_names and align_mapping:
                        old_min = min(align_mapping['min'].keys())
                        old_max = max(align_mapping['max'].keys())
                        new_min = align_mapping['min'][old_min]
                        new_max = align_mapping['max'][old_max]

                        for col in item_names:
                            if col in df.columns:
                                df[col] = df[col].apply(lambda x: self._align_value(x, old_min, old_max, new_min, new_max))

        elif var_type == 'criterion':
            # custom processing for criterion -> other config structure -> maybe adjust later
            pass
        else:
            raise ValueError("var_type must be either 'predictor' or 'criterion'")

        return df

    def _align_value(self, value: float, old_min: float, old_max: float, new_min: float, new_max: float) -> float:
        """
        Aligns a single value based on the old and new scale range.

        Args:
            value: The value to be aligned.
            old_min: The original minimum value in the old scale.
            old_max: The original maximum value in the old scale.
            new_min: The new minimum value in the new scale.
            new_max: The new maximum value in the new scale.

        Returns:
            float: The aligned value.
        """
        if pd.isna(value):
            return value  # Return NaN values as is

        # Scale the value from the old range to the new range
        if old_min == old_max:
            return new_min  # Avoid division by zero if old_min and old_max are the same
        return new_min + ((value - old_min) * (new_max - new_min)) / (old_max - old_min)


    def create_binary_vars_from_categoricals(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        Creates binary predictors from categorical variables in the DataFrame based on the
        configuration provided. The configuration defines which columns to map and the
        corresponding mappings for each dataset.

        Args:
            df_traits: The DataFrame containing the trait data.

        Returns:
            pd.DataFrame: The DataFrame with new binary columns added.
        """
        demographic_cfg = self.fix_cfg["predictors"]["person_level"]["sociodemographics"]

        for entry in demographic_cfg:
            if self.dataset in entry['item_names']:
                item_names = entry['item_names'][self.dataset]
                new_column_name = entry['name']
                category_mappings = entry.get('category_mappings', {}).get(self.dataset, {})
                match item_names:
                    case str(column_name):
                        df_traits[new_column_name] = self._map_column_to_binary(df_traits, column_name, category_mappings)
                    case list(columns):
                        df_traits[new_column_name] = self._map_columns_to_binary(df_traits, columns, category_mappings)
                    case _:
                        raise ValueError("item_names must be either a string or a list of strings")
        print("succes returning ", self.dataset)
        return df_traits

    def _map_column_to_binary(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.Series:
        """
        Maps a single column to a binary column based on the provided mapping.

        Args:
            df: The DataFrame containing the data.
            column: The column name to map.
            mapping: The mapping dictionary for categorical values.

        Returns:
            pd.Series: A binary column derived from the categorical column.
        """
        if df[column].dtype == object:
            return df[column].apply(
                lambda x: max([mapping.get(int(val), 0) for val in x.split(',') if val.isdigit()])
                if pd.notna(x) else 0
            )
        return df[column].map(lambda x: mapping.get(x, 0))

    def _map_columns_to_binary(self, df: pd.DataFrame, columns: list[str], mapping: dict) -> pd.Series:
        """
        Maps multiple columns to a single binary column based on the provided mapping.

        Args:
            df: The DataFrame containing the data.
            columns: The list of column names to map.
            mapping: The mapping dictionary for categorical values.

        Returns:
            pd.Series: A binary column derived from the categorical columns.
        """
        return df[columns].apply(lambda row: max(
            [self._map_column_to_binary(pd.DataFrame({col: [row[col]]}), col, mapping).iloc[0] for col in columns]
        ), axis=1)

    @abstractmethod
    def merge_states(self, df_dct):
        """
        This method merges the files found in self.path_to_raw_data so that the resulting dataframe contains
        all state data for self.dataset. This must be implemented in the subclasses.

        Args:
            df_dct (dict): Dict containing the filenames as keys and the pd.DataFrames as values

        Returns:
            pd.DataFrame: df containing all state variables for a certain dataset
        """
        pass

    def select_state_vars(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns relevant to the specified dataset.

        Args:
            df_states: A pandas DataFrame containing the esm dataset.

        Returns:
            pd.DataFrame: A DataFrame filtered to include only the relevant columns.
        """
        columns_to_be_selected = []

        # predictors
        for esm_var in self.fix_cfg["predictors"]["self_reported_micro_context"]:
            columns_to_be_selected.extend(self.extract_columns(esm_var['item_names']))

        # criterion - state wb items
        trait_pa_columns = self.fix_cfg["criterion"]["states"][self.dataset]["pa"]
        trait_na_columns = self.fix_cfg["criterion"]["states"][self.dataset]["na"]

        columns_to_be_selected.extend(trait_pa_columns + trait_na_columns)
        df_col_filtered = df_states[columns_to_be_selected]
        return df_col_filtered

    # Does this makes sense?
    #@abstractmethod
    #def dataset_specific_esm_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method must be implemented in the subclasses. It describes some dataset specific processing
        to handle dataset-specific differences in the variables or scales.

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
    #    pass



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

    def extract_columns(self, config_data: Union[dict, list, str]) -> list[str]:
        """
        Recursively extracts column names from nested dictionaries.

        Args:
            config_data: A dictionary containing the configuration data.

        Returns:
            list: A list of columns to be selected for the specified dataset.
        """
        columns = []
        for key, value in config_data.items():
            if isinstance(value, dict):
                columns.extend(self.extract_columns(value))
            elif isinstance(value, list):
                if key == self.dataset:
                    columns.extend(value)
            elif isinstance(value, str):
                if key == self.dataset:
                    columns.append(value)
        return columns




