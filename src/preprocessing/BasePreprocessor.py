from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, List

import numpy as np
import pandas as pd

from src.utils.configparser import *
from src.utils.logger import *
from src.utils.timer import *

import re


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
        # self.raw_trait_id_col = self.fix_cfg["person_level"]["other_trait_columns"][0]["item_names"][self.dataset]
        # self.raw_esm_id_col = self.fix_cfg["esm_based"]["other_esm_columns"][0]["item_names"][self.dataset]
        # self.esm_timestamp = self.fix_cfg["esm_based"]["other_esm_columns"][1]["item_names"][self.dataset]
        self.logger = Logger(log_dir=self.var_cfg["general"]["log_dir"], log_file=self.var_cfg["general"]["log_name"])
        self.timer = Timer(self.logger)  # composition
        self.config_parser = ConfigParser().cfg_parser
        self.apply_preprocessing_methods = self.timer._decorator(self.apply_preprocessing_methods)
        self.data = None  # assigned at the end, preprocessed data, if at all

    @property
    def path_to_raw_data(self):
        """Path to the folder containing the raw files for self.dataset."""
        return os.path.join(self.var_cfg["prelimpreprocessor"]["path_to_raw_data"], self.dataset)

    @property
    def raw_trait_id_col(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["person_level"]["other_trait_columns"][0]["item_names"][self.dataset]

    @property
    def raw_esm_id_col(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["esm_based"]["other_esm_columns"][0]["item_names"][self.dataset]

    @property
    def esm_timestamp(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["esm_based"]["other_esm_columns"][1]["item_names"][self.dataset]



    def apply_preprocessing_methods(self):
        """
        This function applies the preprocessing methods

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
            # (self.convert_str_cols_to_list_cols, {'df': None}),  # This may be slow
            (self.dataset_specific_trait_processing, {'df_traits': None}),
            (self.select_columns, {'df': None, 'df_type': "person_level"}),
            (self.sort_dfs, {'df': None, 'df_type': "person_level"}),
            (self.align_scales, {
                'df': None,
                'df_type': "person_level",
                'cat_list': ['personality', 'sociodemographics', 'criterion'],
            }),
            (self.create_binary_vars_from_categoricals, {'df_traits': None}),
        ]
        for method, kwargs in preprocess_steps_traits:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_traits for k, v in kwargs.items()}
            df_traits = self._log_and_execute(method, **kwargs)
        print()

        # Step 3: Process and transform esm data
        preprocess_steps_esm = [
            (self.merge_states, {'df_dct': df_dct}),
            (self.dataset_specific_state_processing, {'df_states': None}),
            (self.select_columns, {'df': None, 'df_type': "esm_based"}),
            (self.sort_dfs, {'df': None, 'df_type': "esm_based"}),
            (self.align_scales, {
                'df': None,
                'df_type': "esm_based",
                'cat_list': ['self_reported_micro_context', 'criterion'],  # sensed microcontext?
            }),
            (self.create_person_level_vars_from_esm, {'df_states': None})
        ]
        for method, kwargs in preprocess_steps_esm:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_states for k, v in kwargs.items()}
            df_states = self._log_and_execute(method, **kwargs)
            print()

        # Step 4: merge data

        self.logger.log(f"Finished preprocessing pipeline for >>>{self.dataset}<<<")
        return df_traits

    def _log_and_execute(self, method: Callable, *args: Any, **kwargs: Any):
        """

        Args:
            method:
            *args:
            **kwargs:

        Returns:

        """
        self.logger.log(f"   Executing {method.__name__}")
        return method(*args, **kwargs)

    def load_data(self, path_to_dataset):
        """
        This method loads all files contained in self.path_to_raw_data and returns a dict containing pd.DataFrames

        Args:
            path_to_dataset:

        Returns:


        """
        files = os.listdir(path_to_dataset)
        if files:
            df_dct = {file[:-4]: pd.read_csv(os.path.join(path_to_dataset, file), encoding="latin", nrows=1000)
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
        flags = self.config_parser(self.fix_cfg["person_level"]["other_trait_columns"], "binary", "trait_flags")[0]
        flag_col = flags["item_names"][self.dataset]
        df_traits[flag_col] = df_traits[flag_col].map(flags["category_mappings"][self.dataset]).fillna(np.nan)
        flag_filtered_df = df_traits[df_traits[flag_col] == 1]
        return flag_filtered_df

    def select_columns(self, df: pd.DataFrame, df_type: str = "person_level") -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns relevant to the specified dataset.

        Args:
            df: A pandas DataFrame containing the dataset.
            df_type: "person_level", or "esm_based"

        Returns:
            pd.DataFrame: A DataFrame filtered to include only the relevant columns.
        """
        cols_to_be_selected = []
        for cat, cat_entries in self.fix_cfg[df_type].items():
            for entry in cat_entries:
                if "item_names" in entry:
                    cols_to_be_selected.extend(self.extract_columns(entry['item_names']))
        df_col_filtered = df[cols_to_be_selected]
        df_col_filtered = df_col_filtered.loc[:, ~df_col_filtered.columns.duplicated()].copy()
        return df_col_filtered

    def sort_dfs(self, df: pd.DataFrame, df_type: str = "person_level") -> pd.DataFrame:
        """
        This methods
            a) the person-level df by id
            b) the esm df by id and timestamp

        Args:
            df:
            df_type: "person_level" or "esm_based"

        Returns:
            pd.DataFrame:
        """
        if df_type == "person_level":
            return df.sort_values(by=[self.raw_trait_id_col])
        elif df_type == "esm_based":
            return df.sort_values(by=[self.raw_esm_id_col, self.esm_timestamp])
        else:
            raise ValueError(f"Wrong df_type {df_type}, needs to be 'person_level' or 'esm_based'")


    def convert_str_cols_to_list_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method converts all columns that contain multi-number strings (sth like "1, 2, 3") to a list of numbers.

        Args:
            df (pd.DataFrame): Input DataFrame to process.

        Returns:
            pd.DataFrame: Processed DataFrame with lists in place of comma-separated strings.
        """
        df_copy = df.copy()  # Copy the original DataFrame for comparison

        for col in df.columns:
            df[col] = df[col].apply(lambda x: self.convert_to_list(x, col))

            # Compare original column with the modified column
            if not df[col].equals(df_copy[col]):
                self.logger.log(f"-----Converted multi-number string to list for {col}")
        return df

    def convert_to_list(self, cell_value: str, column_name: str) -> list[int | str] | str:
        """
        Converts a string of comma-separated values into a list of integers or strings.
        If the values can be converted to integers, they will be; otherwise, they remain strings.
        If a non-digit value is found in a comma-separated string, it logs the occurrence.
        If the cell_value is not a valid string or doesn't contain commas, it returns the original value.
        """
        pattern = r'^(\s*\d+\s*,\s*)*\d+\s*$'  # Number followed by comma

        if isinstance(cell_value, str) and re.match(pattern, cell_value):
            result = []
            for x in cell_value.split(','):
                x = x.strip()
                if x.isdigit():
                    result.append(int(x))
                else:
                    self.logger.log(f"-----Found non-digit value '{x}' in column {column_name}")
                    result.append(x)
            return result
        else:
            return cell_value


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

    def align_scales(self, df: pd.DataFrame, cat_list: list, df_type: str = 'person_level') -> pd.DataFrame:
        """
        Aligns the scales of specified columns in the DataFrame according to the configuration.

        This method searches through the configuration and processes only those entries where
        `self.dataset` has a corresponding key in `align_scales_mapping`. For each such entry,
        it aligns the numerical values of the columns listed in `item_names[self.dataset]` based
        on the scaling mapping provided in `align_scales_mapping[self.dataset]`.

        Args:
            df: The DataFrame containing either trait or esm data.
            df_type: "person_level" or "esm_based"
            cat_list: variable number of feature category strings

        Returns:
            pd.DataFrame: The DataFrame with the aligned scales.
        """
        # TODO: Does this exclude values that are out of the range (e.g. -77, 0)?
        specific_cfg = self.fix_cfg[df_type]
        for cat in cat_list:
            for entry in specific_cfg[cat]:
                if "item_names" in entry:
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
        demographic_cfg = self.fix_cfg["person_level"]["sociodemographics"]

        for entry in demographic_cfg:
            if self.dataset in entry['item_names'] and entry["var_type"] == "binary":
                item_names = entry['item_names'][self.dataset]
                new_column_name = entry['name']
                category_mappings = entry.get('category_mappings', {}).get(self.dataset, {})
                match item_names:
                    case str(column_name):  # TODO: Can I make this more generic? to use with micro context?
                        df_traits[new_column_name] = self._map_column_to_binary(df_traits, column_name, category_mappings)
                    case list(columns):
                        df_traits[new_column_name] = self._map_columns_to_binary(df_traits, columns, category_mappings)
                    case _:
                        raise ValueError("item_names must be either a string or a list of strings")
        print("success returning ", self.dataset)
        return df_traits

    def _map_column_to_binary(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.Series:
        """
        Maps a single column to a binary column based on the provided mapping.
        If a cell contains multiple values separated by commas, the mapping will be applied if any value matches the mapping.

        Args:
            df: The DataFrame containing the data.
            column: The column name to map.
            mapping: The mapping dictionary for categorical values.

        Returns:
            pd.Series: A binary column derived from the categorical column.
        """
        # Check if the column is of type object (comma-separated values) or something else
        if df[column].dtype == object:
            # Apply conversion for object columns
            return df[column].apply(lambda x: self._map_comma_separated(x, mapping))

        # Apply the mapping directly for non-object columns
        return df[column].map(lambda x: self._map_single_value(x, mapping))

    def _map_comma_separated(self, cell_value: str, mapping: dict, map_ambiguous: bool = True) -> Union[int, float]:  # or bool? 0/1?
        """
        Processes cells that may contain comma-separated values or a single value.
        map_ambiguous determines how it handles comma-separated values where one value is contained in the mapping,
        and the others aren't (e.g., "selection_medium" describes the context of a social interaction, which may
        be in-person, but also on phone; the "ftf_interactions" mapping only matches the value for in-person interaction.
        We decided to assign only unambiguous interactions values (1 if only in-person interactions, 0 if only cmc
        interactions). The rest should be set to np.nan, so that the percentage calculations are valid.
        """
        if pd.isna(cell_value):
            return np.nan  # return 0
        # TODO: For map_ambiuous == False, this must return np.nan. Also for the others?

        # Split if comma-separated, map each value, and return the maximum mapped value
        values = cell_value.split(',')
        mapped_values = [self._map_single_value(int(val.strip()), mapping) for val in values if val.strip().isdigit()]

        if not map_ambiguous:
            if len(set(mapped_values)) > 1:
                mapped_values = [np.nan]
        return max(mapped_values)  # , default=0)

    @staticmethod
    def _map_single_value(value, mapping: dict) -> Union[int, float]:
        """
        Maps a single value using the provided mapping dictionary. If the input value is np.nan
        it should return np.nan ("real" missings).
        If the value is a number but not specified in the mapping, it should return zero.
        """
        if pd.isna(value):
            return np.nan
        return mapping.get(value, 0)  # mapping.get(value, 0)

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

    def dataset_specific_state_processing(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method may be adjusted in specific subclasses that need dataset-specific processing
        that applies to special usecases.

        Args:
            df_states:

        Returns:
            pd.DataFrame:
        """
        return df_states

    def create_person_level_vars_from_esm(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This is a wrapper for the different methods to create person level variables from the ESM dataset. This includes
            - calculating percentages of categorical variables (e.g.,  ftf_interactions)
            - calculating descriptive statistics (M, SD, min, max) of continuous variables (e.g., sleep_quality)
            - calculating some custom variables that are included in all dfs (e.g., weekday_responses
        At the end, the df is collapsed to the person-level.

        Args:

        Returns:

        """
        df_states = self.create_person_level_desc_stats(df_states=df_states)
        df_states = self.create_person_level_percentages(df_states=df_states)
        df_states = self.create_weekday_responses(df_states=df_states)
        df_states = self.create_early_day_responses(df_states=df_states)
        df_states = self.create_number_responses(df_states=df_states)
        df_states = self.create_percentage_responses(df_states=df_states)

        df_states_person_level = self.collapse_state_df(df_states=df_states)
        return df_states_person_level

    def create_person_level_desc_stats(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates M, SD, Min and Max for the continuous variables assessed in the ESM surveys
        to include it as predictors on a person-level.

        Args:
            df_states:

        Returns:
            pd.DataFrame

        """
        # Extract the continuous variable entries from the config
        cont_var_entries = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                              "continuous",
                                              "number_interaction_partners",  # only two vars where we compute the descs
                                              "sleep_quality")

        # Group the dataset (df_states) by the person ID column
        grouped_df = df_states.groupby(self.raw_esm_id_col)

        # Loop over each continuous variable entry and calculate statistics
        for entry in cont_var_entries:
            var_name = entry['name']

            # Ensure that the dataset key (self.dataset) exists in item_names
            if self.dataset in entry['item_names']:
                # Get the corresponding column name for this dataset
                column = entry['item_names'][self.dataset]

                # Check if the column exists in df_states
                if column in df_states.columns:
                    # Calculate mean, standard deviation, min, and max for the current column
                    stats = grouped_df[column].agg(
                        mean='mean',
                        sd='std',
                        min='min',
                        max='max',
                    ).reset_index()

                    # Rename the columns with the format "name_mean", "name_sd", etc.
                    stats.columns = [self.raw_esm_id_col,
                                     f"{var_name}_mean",
                                     f"{var_name}_sd",
                                     f"{var_name}_min",
                                     f"{var_name}_max"]

                    # Merge the statistics back into the original df_states
                    df_states = pd.merge(df_states, stats, on=self.raw_esm_id_col, how='left')
                else:
                    raise KeyError(f"Column: {column} not found in {self.dataset} state_df")

        return df_states

    def create_person_level_percentages(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates the ratios for several ESM-based ratios on a person-level (e.g., how many
        ftf_interactions vs. cmc interactions as a percentage). To get there, it uses the dataset-specific
        columns specified in the config under "item_names". We have the following cases we need to match
        column types:
            - comma-seperated values (e.g., selection_medium -> Multiple comma separated values)
            - single numeric values (e.g., social interaction -> yes or no)
        aggregation levels:
            - across rows, grouped by person (e.g., computing the ftf vs cmc percentage for each person across prompts)
            - across days, grouped by person (e.g., computing the % of days infeceted for each person across days)
        All cases are adequately handled by this function.

        Args:
            df_states (pd.DataFrame): The dataframe containing state data.

        Returns:
            pd.DataFrame: The original dataframe with new percentage columns added based on config.
        """
        # Extract config entries where var_type == "percentage"
        percentage_var_entries = self.config_parser(cfg=self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                                    var_type="percentage")
        df_copied = deepcopy(df_states)
        person_level_stats = []

        # Loop over each percentage variable entry
        for entry in percentage_var_entries:
            var_name = entry['name']

            # Ensure that the dataset key (self.dataset) exists in item_names
            if "item_names" in entry and self.dataset in entry['item_names']:
                column = entry['item_names'][self.dataset]

                # If the column is a string (not a list), proceed with processing
                if isinstance(column, str):
                    if column in df_states.columns:
                        df_copied = deepcopy(df_states)

                        # We only need to apply the mapping if there is an entry in category_mapping
                        if 'category_mappings' in entry and self.dataset in entry['category_mappings']:
                            category_mapping = entry['category_mappings'][self.dataset]
                            if df_copied[column].dtype in [object, str]:
                                df_copied[column] = df_copied[column].apply(
                                    lambda x: self._map_comma_separated(x, category_mapping, False)
                                )
                            elif df_copied[column].dtype in [int, float]:
                                df_copied[column] = df_copied[column].apply(
                                    lambda x: self._map_single_value(x, category_mapping)
                                )
                            else:
                                raise ValueError(f"{column} dtype must be object, str, int, or float")

                        # for day-level variables (e.g., days_infected)
                        if "per_day" in entry:
                            df_copied[self.esm_timestamp] = pd.to_datetime(df_copied[self.esm_timestamp]).dt.date

                        # Select only relevant columns for this processing step
                        df_copied = (df_copied[[self.raw_esm_id_col, self.esm_timestamp, column]]
                                     .drop_duplicates()
                                     .sort_values([self.raw_esm_id_col, self.esm_timestamp])
                                     )
                        grouped_df = df_copied.groupby(self.raw_esm_id_col)

                        # Group by person ID and calculate the percentage of 1s for each person
                        stats = grouped_df[column].apply(
                            lambda group: (group == 1).sum() / group.isin([0, 1]).sum()
                        ).reset_index()

                        # Rename the column with the format "name_perc"
                        stats.columns = [self.raw_esm_id_col, var_name]

                        # Append to the list of dataframes
                        person_level_stats.append(stats)

                    else:
                        raise KeyError(f"Column: {column} not found in {self.dataset} state_df")
                else:
                    # Skip the entry if it's a list (as the custom function has already built the variable)
                    continue

        # Merge all the person-level percentage dataframes on the person ID column
        if person_level_stats:
            for stats_df in person_level_stats:
                df_states = pd.merge(df_states, stats_df, on=self.raw_esm_id_col, how='left')

        return df_states

    def create_weekday_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses that occurred on weekdays (Monday to Friday) for each person, grouped by
        the person ID (self.raw_esm_id_col). A new column 'weekday_response_percentage' will be added to the DataFrame
        with the calculated percentage for each person.

        Args:
            df_states (pd.DataFrame): The DataFrame containing responses and timestamp information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'weekday_response_percentage' per person.
        """

        # Convert the esm_timestamp column to datetime if not already
        df_states[self.esm_timestamp] = pd.to_datetime(df_states[self.esm_timestamp], errors='coerce')

        # Create a boolean mask for weekdays (Monday=0, ..., Sunday=6)
        df_states['is_weekday'] = df_states[self.esm_timestamp].dt.weekday < 5  # True for Monday to Friday

        # Group by person ID and calculate the percentage of weekday responses for each person
        weekday_stats = df_states.groupby(self.raw_esm_id_col)['is_weekday'].mean()

        # Merge the weekday percentages back into the original dataframe
        df_states = df_states.merge(weekday_stats.rename('weekday_response'),
                                    on=self.raw_esm_id_col,
                                    how='left')

        # Drop the temporary 'is_weekday' column
        # df_states.drop(columns=['is_weekday'], inplace=True)

        return df_states

    def create_early_day_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses that occurred in the 'early' time window (3 AM to 3 PM), based on a daily
        cutoff at 3 AM. Grouped by the person ID (self.raw_esm_id_col), the method adds a new column
        'early_day_responses' to the DataFrame with the calculated percentage.

        Args:
            df_states (pd.DataFrame): The DataFrame containing responses and timestamp information.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'early_day_responses' per person.
        """

        # Convert the esm_timestamp column to datetime if not already
        df_states[self.esm_timestamp] = pd.to_datetime(df_states[self.esm_timestamp], errors='coerce')

        # Adjust the timestamps to start the day from 3 AM
        df_states['hour_adjusted'] = (df_states[self.esm_timestamp] - pd.DateOffset(hours=3)).dt.hour

        # Create a boolean mask for responses between 3 AM and 3 PM (adjusted)
        df_states['is_early'] = df_states['hour_adjusted'].between(0, 11, inclusive="both")

        # Group by person ID and calculate the percentage of early responses for each person
        early_stats = df_states.groupby(self.raw_esm_id_col)['is_early'].mean()

        # Merge the early day percentages back into the original dataframe
        df_states = df_states.merge(early_stats.rename('early_day_responses'),
                                    on=self.raw_esm_id_col,
                                    how='left')

        # Drop the temporary columns
        # df_states.drop(columns=['is_early', 'hour_adjusted'], inplace=True)

        return df_states

    def create_number_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the total number of responses per person and adds it as a column 'number_of_responses' to the
        DataFrame. Each response is represented by one row.

        Args:
            df_states (pd.DataFrame): The DataFrame containing the responses.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'number_of_responses' per person.
        """

        # Group by person ID and count the number of responses per person
        response_counts = (df_states
                           .groupby(self.raw_esm_id_col)
                           .size()
                           .reset_index(name='number_responses')
                           )

        # Merge the response counts back into the original dataframe
        df_states = df_states.merge(response_counts, on=self.raw_esm_id_col, how='left')

        return df_states

    def create_percentage_responses(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the percentage of responses per person by dividing the actual number of responses by the maximum
        possible responses for the given dataset, based on the special_mappings in the config.

        Args:
            df_states (pd.DataFrame): The DataFrame containing the 'number_responses' column.

        Returns:
            pd.DataFrame: The modified DataFrame with an added column 'percentage_responses' per person.
        """

        max_responses = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                           "percentage", "percentage_responses")[0]["special_mappings"][self.dataset]
        # Check if the 'number_responses' column exists
        if 'number_responses' not in df_states.columns:
            raise KeyError("The column 'number_responses' is missing in the DataFrame.")

        # Calculate the percentage of responses per person by dividing actual responses by max possible responses
        df_states['percentage_responses'] = (df_states['number_responses'] / max_responses)

        return df_states




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





