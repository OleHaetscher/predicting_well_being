import pickle
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from typing import Union

import pandas as pd

from src.utils.ConfigParser import *
from src.utils.DataLoader import DataLoader
from src.utils.Logger import *
from src.utils.sanitychecker import *
from src.utils.Timer import *


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
        self.timer = Timer(self.logger)
        self.config_parser = ConfigParser().cfg_parser
        self.config_key_finder = ConfigParser().find_key_in_config
        self.apply_preprocessing_methods = self.timer._decorator(self.apply_preprocessing_methods)
        self.sanity_checker = SanityChecker(logger=self.logger,
                                            fix_cfg=self.fix_cfg,
                                            cfg_sanity_checks=self.var_cfg["preprocessing"]["sanity_checking"],
                                            config_parser_class=ConfigParser(),
                                            apply_to_full_df=False
                                            )
        self.data_loader = DataLoader(nrows=self.var_cfg["preprocessing"]["nrows"])

        self.df_before_final_selection = None  # filled later
        self.df_states = None
        self.data = None  # assigned at the end, preprocessed data, if at all

    @property
    def path_to_raw_data(self):
        """Path to the folder containing the raw files for self.dataset."""
        return os.path.join(self.var_cfg["preprocessing"]["path_to_raw_data"], self.dataset)

    @property
    def path_to_country_level_data(self):
        """Path to the folder containing the raw files for self.dataset."""
        return os.path.join(self.var_cfg["preprocessing"]["path_to_raw_data"], "country_level_vars")

    @property
    def path_to_sensing_data(self):
        """Path to the folder containing the raw files for self.dataset."""
        if self.dataset in ["cocoms", "zpid"]:  # os.path.join(self.path_to_raw_data, "country_level_vars")
            return os.path.join(self.var_cfg["preprocessing"]["path_to_raw_data"], self.dataset, "sensing_vars")
        else:
            return None

    @property
    def raw_trait_id_col(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["person_level"]["other_trait_columns"][0]["item_names"][self.dataset]

    @property
    def raw_esm_id_col(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["esm_based"]["other_esm_columns"][0]["item_names"][self.dataset]

    @property
    def raw_sensing_id_col(self):
        """Dataset specific sensing ID col, must be instantiated after the super.init of the subclasses."""
        if self.dataset in ["cocoms", "zpid"]:
            return self.fix_cfg["sensing_based"]["phone"][0]["item_names"][self.dataset]
        else:
            return None

    @property
    def esm_timestamp(self):
        """Dataset specific ID col, must be instantiated after the super.init of the subclasses."""
        return self.fix_cfg["esm_based"]["other_esm_columns"][1]["item_names"][self.dataset]

    def apply_preprocessing_methods(self):
        """
        This function applies the preprocessing methods

        Returns:

        """
        self.logger.log(f"--------------------------------------------------------")
        self.logger.log(f".")
        self.logger.log(f"Starting preprocessing pipeline for >>>{self.dataset}<<<")

        # Step 1: Load data
        df_dct = self.data_loader.read_csv(path_to_dataset=self.path_to_raw_data)
        df_traits = None  # Initialize df_traits as None
        df_states = None
        df_joint = None

        # Step 2: Process and transform trait data
        self.logger.log(f".")
        self.logger.log(f"  Preprocess trait-survey-based data")
        preprocess_steps_traits = [
            (self.merge_traits, {'df_dct': df_dct}),
            (self.clean_trait_col_duplicates, {'df_traits': None}),
            (self.exclude_flagged_rows, {'df_traits': None}),
            # (self.convert_str_cols_to_list_cols, {'df': None}),  # This may be slow
            (self.adjust_education_level, {'df_traits': None}),
            (self.dataset_specific_trait_processing, {'df_traits': None}),
            (self.select_columns, {'df': None, 'df_type': "person_level"}),
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

        # Step 3: Process and transform esm data
        self.logger.log(f".")
        self.logger.log(f"  Preprocess esm-based data")
        preprocess_steps_esm = [
            (self.merge_states, {'df_dct': df_dct}),
            (self.dataset_specific_state_processing, {'df_states': None}),
            (self.select_columns, {'df': None, 'df_type': "esm_based"}),
            (self.align_scales, {
                'df': None,
                'df_type': "esm_based",
                'cat_list': ['self_reported_micro_context', 'criterion'],  # sensed microcontext?
            }),
            (self.filter_min_num_esm_measures, {'df_states': None}),  # TODO Check if this changes sample sizes
            (self.store_wb_items, {'df_states': None}),
            (self.create_person_level_vars_from_esm, {'df_states': None}),
            # >>> may store the ESM data at this point or directly compute the descriptive where we need the ESM data <<<
            (self.collapse_df, {'df': None, 'df_type': "esm_based"}),
        ]
        for method, kwargs in preprocess_steps_esm:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_states for k, v in kwargs.items()}
            df_states = self._log_and_execute(method, **kwargs)

        # Step 4: Merge state and trait data
        df_joint = self._log_and_execute(self.merge_dfs_on_id, **{"df_states": df_states, "df_traits": df_traits})

        # Step 4 (optional): Specific sensed data processing
        if self.dataset in ["cocoms", "zpid"]:
            self.logger.log(f".")
            self.logger.log(f"  Preprocess sensed data")
            df_dct = self.data_loader.read_r(path_to_dataset=self.path_to_sensing_data)
            df_joint = self.process_and_merge_sensing_data(sensing_dct=df_dct, df=df_joint)

        # Step 5: Specific country-data processing -> this applies to all datasets
        self.logger.log(f".")
        self.logger.log(f"  Preprocess country-level data")
        df_dct = self.data_loader.read_csv(path_to_dataset=self.path_to_country_level_data)
        df_joint = self.merge_country_data(country_var_dct=df_dct, df=df_joint)

        # Step 4: merge data
        self.logger.log(f".")
        self.logger.log(f"  Preprocess joint data")
        preprocess_steps_joint = [
            # (self.merge_dfs_on_id, {"df_states": df_states, "df_traits": df_traits}),
            (self.dataset_specific_post_processing, {'df': None}),
            (self.set_id_as_index, {'df': None}),
            (self.inverse_coding, {'df': None}),  # this makes sense here
            (self.create_scale_means, {'df': None}),
            (self.create_criteria, {'df': None}),
            (self.set_full_col_df_as_attr, {'df': None}),
            (self.fill_unique_id_col, {'df': None}),
            (self.select_final_columns, {'df': None}),
            (self.sanity_checking, {'df': None}),  # this includes selecting the final columns
        ]

        for method, kwargs in preprocess_steps_joint:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_joint for k, v in kwargs.items()}
            df_joint = self._log_and_execute(method, **kwargs)

        self.logger.log(".")
        self.logger.log(f"Finished preprocessing pipeline for >>>{self.dataset}<<<")
        self.logger.log(".")
        self.logger.log(f"--------------------------------------------------------")

        # return df
        return df_joint

    def _log_and_execute(self, method: Callable, *args: Any, indent: int = 4, **kwargs: Any):
        """
        Logs the execution of a method with a customizable indent.

        Args:
            method: The method to be executed.
            indent: The number of spaces to add before the log message (default is 4).
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution.
        """
        indent_spaces = ' ' * indent  # Create a string with the specified number of spaces
        log_message = f"{indent_spaces}Executing {method.__name__}"
        self.logger.log(log_message)
        print(log_message)

        return method(*args, **kwargs)

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
        return df_traits

    def exclude_flagged_rows(self, df_traits) -> pd.DataFrame:
        """
        This function excludes persons from the trait df that indicated that they did
        responded careless.

        Args:
            df_traits:

        Returns:

        """
        flags = self.config_parser(self.fix_cfg["person_level"]["other_trait_columns"], "binary", "trait_flags")[0]
        if self.dataset in flags["item_names"]:
            flag_col = flags["item_names"][self.dataset]
            df_traits[flag_col] = df_traits[flag_col].map(flags["category_mappings"][self.dataset]).fillna(1)
            self.logger.log(
                f"      Persons in trait_df before excluding flagged samples: {df_traits[self.raw_trait_id_col].nunique()}")
            df_traits = df_traits[df_traits[flag_col] == 1]
            self.logger.log(
                f"      Persons in trait_df after excluding flagged samples: {df_traits[self.raw_trait_id_col].nunique()}")
        return df_traits

    def select_columns(self, df: pd.DataFrame, df_type: str = "person_level") -> pd.DataFrame:
        """
        Filters the DataFrame to include only columns relevant to the specified dataset.

        Args:
            df: A pandas DataFrame containing the dataset.
            df_type: "person_level", "esm_based", or "sensing_based"

        Returns:
            pd.DataFrame: A DataFrame filtered to include only the relevant columns.
        """
        test2 = [i for i in df.columns if "relationship" in i]
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

    def adjust_education_level(self, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This method maps the different scales that assessed the education level to a common scale,
        so that for all samples the individual steps of the scale have the same meaning.
        This corresponds to
            1: no education level
            2: primary education level / Hauptschule
            3: lower secondary eduaction / Mittlere Reife
            4: A-Level / Abitur oder Fachhochschulreife
            5: Degree from University or FH
            6: Phd / Promotion

        Args:
            df_traits:

        Returns:
            pd.DataFrame
        """
        education_cfg = self.config_parser(self.fix_cfg["person_level"]["sociodemographics"],
                                           "continuous",
                                           "education_level")[0]
        col = education_cfg["item_names"][self.dataset]
        if self.dataset in education_cfg["category_mappings"]:
            mapping = education_cfg["category_mappings"][self.dataset]
            df_traits[col] = df_traits[col].map(mapping)
        return df_traits

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
                        if isinstance(item_names, str):
                            item_names = [item_names]
                        old_min = min(align_mapping['min'].keys())
                        old_max = max(align_mapping['max'].keys())
                        new_min = align_mapping['min'][old_min]
                        new_max = align_mapping['max'][old_max]

                        for col in item_names:
                            df[col] = df[col].apply(lambda x: self._align_value(x, old_min, old_max, new_min, new_max))
                            self.logger.log(f"        align scales executed for column {col}")
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

        if isinstance(value, str):
            value = pd.to_numeric(value, errors='coerce')

        # Scale the value from the old range to the new range
        if old_min == old_max:
            return new_min  # Avoid division by zero if old_min and old_max are the same
        new_val = new_min + ((value - old_min) * (new_max - new_min)) / (old_max - old_min)
        return new_val

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
                    case str(column_name):
                        df_traits[new_column_name] = self._map_column_to_binary(df=df_traits,
                                                                                column=column_name,
                                                                                mapping=category_mappings,
                                                                                fill_na_with_zeros=False)
                        self.logger.log(f"        Created {new_column_name} from {column_name}")
                    case list(columns):
                        df_traits[new_column_name] = self._map_columns_to_binary(df=df_traits,
                                                                                 columns=columns,
                                                                                 mapping=category_mappings,
                                                                                 fill_na_with_zeros=False)
                        self.logger.log(f"        Created {new_column_name} from {columns}")
                    case _:
                        # raise ValueError("item_names must be either a string or a list of strings")
                        self.logger.log(f" WARNING: {item_names} are neither of type str nor of type list, skip")
                        continue
        return df_traits

    def _map_column_to_binary(self, df: pd.DataFrame, column: str, mapping: dict, fill_na_with_zeros: bool = False) -> pd.Series:
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
            return df[column].apply(lambda x: self._map_comma_separated(cell_value=x,
                                                                        mapping=mapping,
                                                                        map_ambiguous=True,
                                                                        fill_na_with_zeros=fill_na_with_zeros))

        # Apply the mapping directly for non-object columns
        return df[column].map(lambda x: self._map_single_value(cell_value=x, mapping=mapping, fill_na_with_zeros=fill_na_with_zeros))

    def _map_comma_separated(self,
                             cell_value: str,
                             mapping: dict,
                             map_ambiguous: bool = True,
                             fill_na_with_zeros: bool = False) -> Union[int, float]:  # or bool? 0/1?
        """
        Processes cells that may contain comma-separated values or a single value.
        map_ambiguous determines how it handles comma-separated values where one value is contained in the mapping,
        and the others aren't (e.g., "selection_medium" describes the context of a social interaction, which may
        be in-person, but also on phone; the "ftf_interactions" mapping only matches the value for in-person interaction.
        We decided to assign only unambiguous interactions values (1 if only in-person interactions, 0 if only cmc
        interactions). The rest should be set to np.nan, so that the percentage calculations are valid.
        """
        if pd.isna(cell_value):
            if fill_na_with_zeros:
                return 0
            else:
                return np.nan  # return 0

        # Split if comma-separated, map each value, and return the maximum mapped value
        values = cell_value.split(',')
        mapped_values = [self._map_single_value(int(val.strip()), mapping) for val in values if val.strip().isdigit()]

        if not map_ambiguous:
            if len(set(mapped_values)) > 1:
                mapped_values = [np.nan]  # thats stil valid
        return max(mapped_values)

    @staticmethod
    def _map_single_value(cell_value, mapping: dict, fill_na_with_zeros: bool = False) -> Union[int, float]:
        """
        Maps a single value using the provided mapping dictionary. If the input value is np.nan
        it should return np.nan ("real" missings).
        If the value is a number but not specified in the mapping, it should return zero.
        """
        if pd.isna(cell_value):
            if fill_na_with_zeros:
                return 0
            else:
                return np.nan  # return 0
        return mapping.get(cell_value, 0)  # mapping.get(value, 0)

    def _map_columns_to_binary(self, df: pd.DataFrame, columns: list[str], mapping: dict, fill_na_with_zeros: bool = False) -> pd.Series:
        """
        Maps multiple columns to a single binary column based on the provided mapping.

        Args:
            df: The DataFrame containing the data.
            columns: The list of column names to map.
            mapping: The mapping dictionary for categorical values.

        Returns:
            pd.Series: A binary column derived from the categorical columns.
        """
        return df.apply(lambda row: max([
            self._map_column_to_binary(pd.DataFrame({col: [row[col]]}), col, mapping, fill_na_with_zeros).iloc[0]
            for col in columns
        ]), axis=1)

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

        steps = [
            (self.create_person_level_desc_stats, {'df': None, 'feature_category': "self_reported_micro_context"}),
            (self.create_person_level_percentages, {'df_states': None}),
            (self.create_number_interactions, {'df_states': None}),
            (self.create_weekday_responses, {'df_states': None}),
            (self.create_early_day_responses, {'df_states': None}),
            (self.create_number_responses, {'df_states': None}),
            (self.create_percentage_responses, {'df_states': None}),
            (self.create_years_of_participation, {'df_states': None}),
            (self.average_criterion_items, {'df_states': None}),
        ]

        for method, kwargs in steps:
            # Ensure df_traits is passed as needed
            kwargs = {k: v if v is not None else df_states for k, v in kwargs.items()}
            df_states = self._log_and_execute(method, indent=6, **kwargs)

        return df_states

    def create_person_level_desc_stats(self, df: pd.DataFrame, feature_category: str,) -> pd.DataFrame:
        """
        This method creates M, SD, Min and Max for the continuous variables assessed in the ESM surveys
        to include it as predictors on a person-level.
        It also creates the same statistics for the sensing variables.
        We presuppose at least 2 non-Nan values for creating SD, Min, and Max.

        Args:
            df: df
            feature_category: "self_reported_micro_context" or "sensing_based"

        Returns:
            pd.DataFrame

        """
        if feature_category == "self_reported_micro_context":
            cont_var_entries = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                                  "continuous",
                                                  "number_interaction_partners",
                                                  "sleep_quality")
            id_col = self.raw_esm_id_col
        elif feature_category == "sensing_based":
            phone_entries = self.config_parser(self.fix_cfg["sensing_based"]["phone"], "continuous")
            gps_weather_entries = self.config_parser(self.fix_cfg["sensing_based"]["gps_weather"], "continuous")
            cont_var_entries = phone_entries + gps_weather_entries
            id_col = self.raw_sensing_id_col
        else:
            raise ValueError("Feature category must be 'self_reported_micro_context' or 'sensing_based'")

        # Group the dataset (df) by the person ID column
        grouped_df = df.groupby(id_col)

        # Loop over each continuous variable entry and calculate statistics
        for entry in cont_var_entries:
            var_name = entry['name']
            # Ensure that the dataset key (self.dataset) exists in item_names
            if self.dataset in entry['item_names']:
                # Get the corresponding column name for this dataset
                column = entry['item_names'][self.dataset]

                # Check if the column exists in df
                if column in df.columns:
                    # Filter out groups where less than 2 non-NaN values are present for each group
                    valid_counts = grouped_df[column].count().reset_index(name='count')

                    # Filter out groups with less than 2 non-NaN values
                    valid_groups = valid_counts[valid_counts['count'] >= 2][id_col]

                    # Filter the original dataframe to keep only valid groups
                    filtered_df = df[df[id_col].isin(valid_groups)]

                    # Group the filtered dataframe
                    grouped_filtered_df = filtered_df.groupby(id_col)

                    # Calculate mean, standard deviation, min, and max for the current column
                    stats = grouped_filtered_df[column].agg(
                        mean='mean',
                        sd=lambda x: x.std() if x.count() >= 2 else pd.NA,
                        min=lambda x: x.min() if x.count() >= 2 else pd.NA,
                        max=lambda x: x.max() if x.count() >= 2 else pd.NA,
                    ).reset_index()

                    # Rename the columns with the format "name_mean", "name_sd", etc.
                    stats.columns = [id_col,
                                     f"{var_name}_mean",
                                     f"{var_name}_sd",
                                     f"{var_name}_min",
                                     f"{var_name}_max"]

                    self.logger.log(f"          Created M, SD, Min, and Max for var {var_name}")

                    # Merge the statistics back into the original df
                    df = pd.merge(df, stats, on=id_col, how='left')
                    # Drop base column
                    df = df.drop(column, axis=1)
                else:
                    raise KeyError(f"Column: {column} not found in {self.dataset} state_df")

        return df

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

                        # for day-level variables (e.g., home_office)
                        if "per_day" in entry:
                            df_copied[self.esm_timestamp] = pd.to_datetime(df_copied[self.esm_timestamp],
                                                                           errors="coerce").dt.date

                        # Select only relevant columns for this processing step
                        df_copied = (df_copied[[self.raw_esm_id_col, self.esm_timestamp, column]]
                                     .drop_duplicates()
                                     .sort_values([self.raw_esm_id_col, self.esm_timestamp])
                                     )
                        grouped_df = df_copied.groupby(self.raw_esm_id_col)

                        # Group by person ID and calculate the percentage of 1s for each person
                        # If no valid values are found np.nan is assigned (also results in RuntimeWarning)
                        stats = grouped_df[column].apply(
                            lambda group: (group == 1).sum() / group.isin([0, 1]).sum()
                        ).reset_index()

                        stats.columns = [self.raw_esm_id_col, f"{var_name}"]
                        person_level_stats.append(stats)

                        self.logger.log(f"          Created Percentage for var {var_name}")

                        assert len(stats[stats[var_name] > 1]) == 0, "percentage found that is greater than 1"

                    else:
                        raise KeyError(f"Column: {column} not found in {self.dataset} state_df")
                else:
                    self.logger.log(f"      Skipping var {var_name} in dataset {self.dataset}, has custom method")
                    print(f"Skipping var {var_name} in dataset {self.dataset}, has custom method")
                    continue

        # Merge all the person-level percentage dataframes on the person ID column
        if person_level_stats:
            for stats_df in person_level_stats:
                df_states = pd.merge(df_states, stats_df, on=self.raw_esm_id_col, how='left')

        return df_states

    def create_number_interactions(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates the person-level variable "number_interactions". Therefore, it sums
        all rows per person where they indicated they had a social interaction. This column is then
        added to the df. THe columns per dataset are given by the config.
        Importantly, this gets overriden in CoCoMS for a custom implementation

        Args:
            df_states:

        Returns:

        """
        cfg_num_ia = self.config_parser(self.fix_cfg["esm_based"]["self_reported_micro_context"],
                                           "continuous",
                                           "number_interactions")[0]
        item_name = cfg_num_ia["item_names"][self.dataset]
        cat_mappings = cfg_num_ia["category_mappings"][self.dataset]

        test = df_states[item_name].value_counts()

        # Mapping the category values for the single item and summing interactions
        df_states["interaction_sum"] = df_states[item_name].map(
            lambda x: cat_mappings.get(x, 0)
        )

        # Grouping by the person ID and summing interactions per person
        df_grouped = df_states.groupby(self.raw_esm_id_col, as_index=False).agg(
            number_interactions=('interaction_sum', 'sum')
        )

        # Merging the result back to the original DataFrame
        df_states = df_states.merge(df_grouped, on=self.raw_esm_id_col, how='left')

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
        df_states = df_states.merge(weekday_stats.rename('weekday_responses'),
                                    on=self.raw_esm_id_col,
                                    how='left')

        # Drop the temporary 'is_weekday' column
        # df_states.drop(columns=['is_weekday'], inplace=True)

        assert len(df_states[df_states['weekday_responses'] > 1]) == 0, "percentage found that is greater than 1"

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

        assert len(df_states[df_states['early_day_responses'] > 1]) == 0, "percentage found that is greater than 1"

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
        study_wave_col = self.config_parser(self.fix_cfg["esm_based"]["other_esm_columns"],
                                           "string", "studyWave")[0]["item_names"]

        # Adjust the maximum for persons that participated in two bursts (if needed)

        if self.dataset in ["emotions", "zpid"]:  # need to adjust for two bursts
            # Multiply max_responses by 2 for those who participated in both waves
            df_states['percentage_responses'] = df_states.apply(
                lambda row: row['number_responses'] / (max_responses * 2) if row[study_wave_col[self.dataset]] == "Both"
                else row['number_responses'] / max_responses,
                axis=1
            )
        else:  # no adjustment necessary
            # Just divide by max_responses
            df_states['percentage_responses'] = df_states['number_responses'] / max_responses

        max_crit_resp = df_states[df_states['percentage_responses'] > 1]
        if len(max_crit_resp) > 0:
            self.logger.log(f"        WARNING: Found values over 1, Max value: {max_crit_resp['percentage_responses'].max()}, set to 1")

        df_states["percentage_responses"] = df_states["percentage_responses"].apply(
            lambda x: 1 if x > 1 else x
        )
        return df_states

    def create_years_of_participation(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        This method creates a column that contains a list of years in corresponding to the years in which
        participants answered ESM surveys. If they participated only in one year, it contains only one item,
        if they participated in two years, in contains to items.

        Args:
            df_states:

        Returns:
            pd.DataFrame: The DataFrame with an additional column "years_of_participation", containing the list of unique years
                        in which each participant participated.
        """
        # Extract the year from the timestamp column
        df_states['year'] = pd.to_datetime(df_states[self.esm_timestamp]).dt.year

        # Group by person_id and collect unique years of participation for each person
        df_years = df_states.groupby(self.raw_esm_id_col)['year'].apply(lambda x: tuple(sorted(x.unique()))).reset_index()

        # Rename the column to 'years_of_participation'
        df_years = df_years.rename(columns={'year': 'years_of_participation'})

        # Merge the years of participation back into the original DataFrame on the person_id column
        df_states = df_states.merge(df_years, on=self.raw_esm_id_col, how='left')

        # Drop the temporary 'year' column, as it's no longer needed
        df_states = df_states.drop(columns=['year'])

        return df_states

    def average_criterion_items(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        In this method, we compute averages of the items assessing positive and negative
        state effect to generate person-level scores of these items

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        affect_states_dct = self.get_state_affect_dct()

        # Iterate over affect traits (positive and negative)
        for affect_state, affect_cols in affect_states_dct.items():
            # Extract relevant item names for the current dataset from the configuration
            if affect_cols:
                # Compute mean for each person and assign it back to the original columns
                person_avg = df_states.groupby(self.raw_esm_id_col)[affect_cols].transform('mean')
                df_states[affect_cols] = person_avg

        return df_states

    def get_state_affect_dct(self) -> dict[str, list]:
        """
        Because this is used multiple times, this helper function returns a dict that contains the type of affect as keys
        (pa, na) and the items in the specific dataset as values.

        Returns:
            dict[str, list]
        """
        # Extract the esm_based criterion variables from the config
        affect_var_entries = self.config_parser(self.fix_cfg["esm_based"]["criterion"],
                                                "continuous",
                                                "pa_state",
                                                "na_state")
        # Extract the relevant variables for both positive and negative affect states
        affect_states_dct = {val["name"]: val["item_names"][self.dataset] for val in affect_var_entries
                             if self.dataset in val["item_names"]}
        return affect_states_dct

    def filter_min_num_esm_measures(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        We only include participants who provided at least 5 ESM measurements with non-missing data on the
        momentary well-being indicators. For the paper, we require 5 measurements

        Args:
            df_states: df containing the states of a given ESM-sample

        Returns:
            filtered_df: Filtered state df
        """
        affect_states_dct = self.get_state_affect_dct()
        min_num_esm = self.var_cfg["preprocessing"]["min_num_esm_measures"]
        pos_affect_cols = affect_states_dct.get('pa_state', [])
        neg_affect_cols = affect_states_dct.get('na_state', [])

        # Filter rows where at least one positive and one negative affect item is not NaN
        if pos_affect_cols and neg_affect_cols:
            # Both positive and negative affect columns are non-empty
            valid_measurements = df_states[pos_affect_cols].notna().any(axis=1) & df_states[
                neg_affect_cols].notna().any(axis=1)
        elif pos_affect_cols:
            # Only positive affect columns exist
            valid_measurements = df_states[pos_affect_cols].notna().any(axis=1)
        elif neg_affect_cols:
            # Only negative affect columns exist
            valid_measurements = df_states[neg_affect_cols].notna().any(axis=1)
        else:
            raise ValueError(f"No positive or negative items found in config for {self.dataset}")

        # Count the number of valid rows per participant
        valid_count_per_person = df_states[valid_measurements].groupby(self.raw_esm_id_col).size()

        # Filter participants who meet the minimum number of valid measurements
        filtered_df = df_states[
            df_states[self.raw_esm_id_col].isin(valid_count_per_person[valid_count_per_person >= min_num_esm].index)
        ]
        persons_in_unfiltered_df = df_states[self.raw_esm_id_col].nunique()
        persons_in_filtered_df = filtered_df[self.raw_esm_id_col].nunique()
        self.logger.log(f"        N persons included in before filtering: {persons_in_unfiltered_df}")
        self.logger.log(f"        N measurements included in before filtering: {len(df_states)}")
        self.logger.log(f"        N persons after require at least {min_num_esm} measurements per person: {persons_in_filtered_df}")
        self.logger.log(f"        N measurements after require at least {min_num_esm} measurements per person: {len(filtered_df)}")

        if self.dataset in ["cocout", "cocoms"]:
            for wave in filtered_df["studyWave"].unique():
                df_filtered_tmp = filtered_df[filtered_df["studyWave"] == wave]
                df_unfiltered_tmp = df_states[df_states["studyWave"] == wave]
                self.logger.log(f"        Split up filtered num measurements included for {self.dataset}")
                self.logger.log(f"          N persons for wave {wave} before filtering: {df_unfiltered_tmp[self.raw_esm_id_col].nunique()}")
                self.logger.log(f"          N measurements for wave {wave} before filtering: {len(df_unfiltered_tmp)}")
                self.logger.log(f"          N persons for wave {wave} after filtering: {df_filtered_tmp[self.raw_esm_id_col].nunique()}")
                self.logger.log(f"          N measurements for wave {wave} after filtering: {len(df_filtered_tmp)}")
        return filtered_df

    def store_wb_items(self, df_states: pd.DataFrame) -> pd.DataFrame:
        """
        With this method, we store the wb-items from the state df for each sample before joining and collapsing the
        dfs, so that we can compute descriptivies statistics for these items in the Postprocessing class.
        It returns df_states though, so that it fits in the preprocessing logic

        Args:
            df_states:

        Returns:
            pd.DataFrame
        """
        cols = [self.raw_esm_id_col]
        affect_states_dct = self.get_state_affect_dct()
        cols.extend([item for sublist in affect_states_dct.values() for item in sublist])
        df_wb_items = df_states[cols]

        pa_items = affect_states_dct["pa_state"]

        if self.dataset == "zpid":
            df_wb_items["state_wb"] = df_wb_items[pa_items[0]]  # valence
        else:
            # create pa / na / wb score
            na_items = affect_states_dct["na_state"]
            df_wb_items[f'state_pa'] = df_wb_items[pa_items].mean(axis=1)
            df_wb_items[f'state_na'] = df_wb_items[na_items].mean(axis=1)

            # Create wb score
            df_wb_items[f'state_na_inv'] = self.inverse_code(df_wb_items[na_items],
                                                             min_scale=1,
                                                             max_scale=6).mean(axis=1)
            new_columns = pd.DataFrame({
                f'state_wb': df_wb_items[[f'state_pa', f'state_na_inv']].mean(axis=1)
            })
            df_wb_items = pd.concat([df_wb_items, new_columns], axis=1)
            df_wb_items = df_wb_items.drop(['state_na_inv'], axis=1)

        filename = os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"],
                                f"wb_items_{self.dataset}")
        with open(filename, "wb") as f:
            pickle.dump(df_wb_items, f)

        return df_states

    def set_full_col_df_as_attr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method sets the state_df as a class attribute before collapsing it, so that we can refer to
        it later when we do the sanity checks (e.g., for calculating scale reliabilities).

        Args:
            df:

        Returns:
            None
        """
        self.df_before_final_selection = deepcopy(df)
        return df

    def dataset_specific_post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Could use this too handle the np.nan / 0 issue more elegant
        """
        Logic depends on the subclass
        Args:
            df:

        Returns:

        """
        return df

    def collapse_df(self, df: pd.DataFrame, df_type: str) -> pd.DataFrame:
        """
        This method transforms the df (that varies by person and esm measurement or date) into a person-level df
        (that only differs by person, one row per person). Therefore, it removes all variables that vary in-person.

        Args:
            df:
            df_type: "esm_based" or "sensing_based"

        Returns:
            pd.DataFrame: A person-level df containing the aggregated information from the ESM-surveys.

        """
        if df_type == "esm_based":
            id_col = self.raw_esm_id_col
        elif df_type == "sensing_based":
            id_col = self.raw_sensing_id_col
        else:
            raise ValueError("df_type must be esm_based or sensing_based")

        constant_cols = []
        for col in df.columns:
            # Check if the column has only one unique value per person
            if df.groupby(id_col)[col].nunique().max() == 1:
                constant_cols.append(col)

        # Select only the constant columns and drop duplicates (one row per person)
        df_person_level = df[constant_cols].drop_duplicates(subset=id_col)
        return df_person_level

    def set_id_as_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new ID column in the form of 'self.dataset_<integer>' and sets it as the index
        of the DataFrame. The index will be unique for each row.

        Args:
            df (pd.DataFrame): The input DataFrame to set the new index for.

        Returns:
            pd.DataFrame: The DataFrame with the new ID column set as the index.
        """
        # Create the new ID column by combining the dataset name with an integer index
        df['new_id'] = self.dataset + "_" + (df.reset_index().index + 1).astype(str)
        # Set the new_id as the index
        df = df.set_index('new_id', drop=True)
        return df

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

    def merge_country_data(self, country_var_dct: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
        """
        This method:
            a) merges the 3 country-level var DataFrames (health, psycho-political, socio-economic),
            b) merges the resulting DataFrame with df on country and the averaged year of participation.
            If participants' assessments span multiple years, the country-level variables are averaged across those years.

        Args:
            country_var_dct: Dictionary containing country-level DataFrames keyed by variable type.
            df: DataFrame containing participant-level data with a 'years_of_participation' column (list of years).

        Returns:
            pd.DataFrame: Merged DataFrame with country-level variables merged into participant data.
        """
        # Step 1: Merge the country-level DataFrames
        df_country_level = reduce(
            lambda left, right: pd.merge(left, right, on=["country", "year"], how="outer"),
            country_var_dct.values()
        )

        df_country_level["democracy_index"] = pd.to_numeric(df_country_level["democracy_index"].str.replace(',', '.', regex=False), errors='coerce')
        country_level_cols_to_agg = df_country_level.columns.drop(["country", "year"])

        # Step 2: Handle multiple years of participation by exploding and merging on the "year" column
        df = df.explode("years_of_participation").rename(columns={"years_of_participation": "year"})

        # Merge participant data with country-level data
        df = pd.merge(df, df_country_level, on=["country", "year"], how="left")

        # Group by participant and country, and apply mean to country-level columns, keeping other columns unchanged
        df = df.groupby([self.raw_esm_id_col, "country"], as_index=False).agg(
            {**{col: 'mean' for col in country_level_cols_to_agg},
             **{col: 'first' for col in df.columns.difference(country_level_cols_to_agg)}}
        )

        return df

    def merge_dfs_on_id(self, df_states: pd.DataFrame, df_traits: pd.DataFrame) -> pd.DataFrame:
        """
        This function merges a variable number of dataframes into a single dataframe.
        The merging is done along the columns, ensuring that the rows remain aligned,
        and the DataFrames are merged on common columns like 'country' and 'year'.

        Args:
            df_states:
            df_traits

        Returns:
            pd.DataFrame: A merged DataFrame along the columns.
        """
        df_joint = pd.merge(df_states, df_traits, left_on=self.raw_esm_id_col, right_on=self.raw_trait_id_col, how="left")
        return df_joint

    def inverse_coding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recodes the items in the DataFrame based on the configuration entries for each variable.
        Only items that need recoding for the specific dataset (self.dataset) are processed, and
        the recoding is done using the provided scale endpoints (min and max).
        Importantly, the differing scales need to be aligned before this processing step.

        Args:
            df (pd.DataFrame): The DataFrame containing the items to be recoded.

        Returns:
            pd.DataFrame: DataFrame with recoded items (where applicable).
        """
        cont_pers_entries = self.config_parser(self.fix_cfg["person_level"]["personality"],
                                               "continuous")
        # if self.dataset != "cocoesm":  # TODO Change again already recoded
        for entry in cont_pers_entries:
            # Check if the entry has "items_to_recode" and if there are items to recode for the given dataset
            if "items_to_recode" in entry and entry["items_to_recode"].get(self.dataset):
                items_to_recode = entry["items_to_recode"][self.dataset]  # Items to be recoded for this dataset
                scale_min = entry["scale_endpoints"]["min"]
                scale_max = entry["scale_endpoints"]["max"]

                # Perform inverse recoding: new_value = scale_max + scale_min - old_value
                for item in items_to_recode:
                    df[item] = pd.to_numeric(df[item])
                    if item in df.columns:  # Ensure the column exists in the DataFrame
                        df[item] = scale_max + scale_min - df[item]
                    else:
                        raise KeyError(f"col {item} not found in {self.dataset} df")
        return df

    def create_scale_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates scale means based on the item columns specified in the config entries for the given dataset.
        A new column will be created for each entry, named after the 'name' key, and its value will be the
        mean of the corresponding item columns for the dataset.

        Args:
            df (pd.DataFrame): The DataFrame containing the items.

        Returns:
            pd.DataFrame: The DataFrame with new columns for the scale means.
        """
        cont_pers_entries = self.config_parser(self.fix_cfg["person_level"]["personality"],
                                               "continuous")
        cont_soc_dem_entries = self.config_parser(self.fix_cfg["person_level"]["sociodemographics"],
                                               "continuous")
        for entry in cont_pers_entries + cont_soc_dem_entries:
            # Get the item names for the current dataset (self.dataset)
            if self.dataset in entry["item_names"]:
                item_cols = entry["item_names"][self.dataset]
                # Create a new column with the name from 'name' and the mean of the valid item columns
                df[item_cols] = df[item_cols].apply(pd.to_numeric, errors='coerce')
                try:
                    # df[entry['name']] = df[item_cols].mean(axis=1, skipna=True)
                    df = df.assign(**{
                        entry['name']: df[item_cols].mean(axis=1, skipna=True)
                    })
                except ValueError:
                    df[entry['name']] = df[item_cols]
        return df

    def create_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to aggregate the items assessing trait and state wb to create the 6 different criteria.
            - state_wb
            - state_pa
            - state_na
            - trait_wb
            - trait_pa
            - trait_na
        To do so, we average across the items to create the pa and na scores, and we inverse code the na score and then
        average the two composite scores to create the wb score

        Args:
            df (pd.DataFrame): The input dataframe containing well-being item data for individuals.

        Returns:
            pd.DataFrame: DataFrame with additional columns for the calculated criteria.
        """
        # TODO: A bit messy, may be improved, but handles all edgecases (emotions, zpid)
        criteria_types = {'trait': "person_level", 'state': "esm_based"}
        # Loop over trait and state criteria
        for criterion_type, config_lvl in criteria_types.items():

            wb_items = self.config_parser(self.fix_cfg[config_lvl]["criterion"], var_type=None)
            pa_items = None
            na_items = None

            if self.dataset == "emotions" and criterion_type == "trait":
                continue

            # pa
            if self.dataset in wb_items[0]["item_names"]:
                pa_items = wb_items[0]["item_names"][self.dataset]
                df[f'{criterion_type}_pa'] = df[pa_items].mean(axis=1)

            # na
            if self.dataset in wb_items[1]["item_names"]:
                na_items = wb_items[1]["item_names"][self.dataset]
                df[f'{criterion_type}_na'] = df[na_items].mean(axis=1)

            # wb
            scale_min = wb_items[1]["scale_endpoints"]["min"]
            scale_max = wb_items[1]["scale_endpoints"]["max"]
            if na_items:
                df[f'{criterion_type}_na_tmp'] = self.inverse_code(df[na_items], min_scale=scale_min, max_scale=scale_max).mean(axis=1)
                # Create a DataFrame with all new columns at once
                new_columns = pd.DataFrame({
                    f'{criterion_type}_wb': df[[f'{criterion_type}_pa', f'{criterion_type}_na_tmp']].mean(axis=1)
                })
                df = pd.concat([df, new_columns], axis=1)
                df = df.drop([f'{criterion_type}_na_tmp'], axis=1)
            else:  # zpid
                df[f'{criterion_type}_wb'] = df[f'{criterion_type}_pa']
                df = df.drop(f'{criterion_type}_pa', axis=1)
        return df

    @staticmethod
    def inverse_code(df: pd.DataFrame, min_scale: int, max_scale: int) -> pd.DataFrame:
        """
        Inverse codes the negative affect items by subtracting each value from the maximum value of the scale.

        Args:
            df (pd.DataFrame): The DataFrame containing the negative affect items.
            min_scale (int): The minimum value of the scale.
            max_scale (int): The maximum value of the scale.

        Returns:
            pd.DataFrame: DataFrame with inverse-coded negative affect items.
        """
        return max_scale + min_scale - df

    def sanity_checking(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function executes several sanity checks using the Sa
        Args:
            df:

        Returns:

        """
        self.sanity_checker.run_sanity_checks(df=df, dataset=self.dataset, df_before_final_sel=self.df_before_final_selection)
        return df

    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function selects the final columns that we include in the final dataframe. This includes
            - person-level variables (pl)
            - self-reported micro context (srmc)
            # - sensed micro context (sensed) #> later
            - macro context (macro)
        It further adds a prefix so that we can more easy identify the columns

        Args:
            df: DataFrame containing the data with a subset of columns from different categories.

        Returns:
            pd.DataFrame: A filtered DataFrame with only the selected columns and prefixes added.
        """
        final_df = pd.DataFrame()

        for prefix, columns in self.fix_cfg["var_assignments"].items():
            selected_columns = [col for col in columns if col in df.columns]
            renamed_columns = {col: f"{prefix}_{col}" for col in selected_columns}
            try:
                prefixed_df = df[selected_columns].rename(columns=renamed_columns)
                final_df = pd.concat([final_df, prefixed_df], axis=1)
            except KeyError:
                print(f"  Some columns of {selected_columns} are not present in {self.dataset} df")
        return final_df

    def fill_unique_id_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function creates a column "unique_id" in all datasets except cocoms and sets the dataframes
        indices as values (thus, except in cocoms, all values in "unique_id" are unique)
        This will allow us to use "GroupKFOld" consistently in the cv procedure so that it affects only cocoms.

        Args:
            df:

        Returns:
            pd.DataFrame
        """
        if "unique_id" in df.columns:
            df["unique_id"] = df["unique_id"].fillna(df.index.to_series())
        else:
            df["unique_id"] = df.index

        test = df.unique_id.isna().sum()
        return df


    def process_and_merge_sensing_data(self, sensing_dct: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
        """
        This method

            - merges the two sensing datasets
            - creates person-level variables from the sensing variables
            - does some sanity checks

        The weather data already corresponds to the individual ESM-period

        Dataset specific sensing processing may include
          - set np.nan to 0 for Apps that were never used
          - set probably 0 to np.nan if our intuition was right
          - apply cut-offs

        Args:
            sensing_dct:
            df:

        Returns:
            pd.DataFrame
        """
        df_sensing = None

        steps = [
            (self.merge_sensing_dfs, {'sensing_dct': sensing_dct}),
            (self.select_columns, {'df': None, 'df_type': "sensing_based"}),
            (self.dataset_specific_sensing_processing, {'df_sensing': None, }),
            (self.change_datetime_to_minutes, {'df': None, "col1": "daily_sunset", "col2": "daily_sunrise"}),
            # TODO: Fix time issues with firstScreen / lastScreen time alignment -> see jupyter
            (self.apply_cut_offs, {'df': None}),
            (self.create_person_level_desc_stats, {'df': None, 'feature_category': "sensing_based"}),
            (self.collapse_df, {'df': None, "df_type": "sensing_based"}),
        ]

        for method, kwargs in steps:
            kwargs = {k: v if v is not None else df_sensing for k, v in kwargs.items()}
            df_sensing = self._log_and_execute(method, indent=6, **kwargs)

        # merge traits
        df = self.merge_state_df_sensing(df=df, df_sensing=df_sensing)

        return df

    def dataset_specific_sensing_processing(self, df_sensing: pd.DataFrame) -> pd.DataFrame:
        """
        Overridden in the subclasses

        Args:
            df_sensing:

        Returns:

        """
        return df_sensing

    def merge_sensing_dfs(self, sensing_dct: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        This method concatenates the two dataframes containing sensing variables (Mobile phone features, GPS + Weather feature)
        along the columns using the "ID" and the "day" column and returns the merged df

        Args:
            sensing_dct:

        Returns:
            pd.DataFrame
        """
        date_col_phone = self.fix_cfg["sensing_based"]["phone"][1]["item_names"][self.dataset]
        date_col_gps_weather = self.fix_cfg["sensing_based"]["gps_weather"][1]["item_names"][self.dataset]

        # Access the respective DataFrames from the dictionary
        phone_df = sensing_dct['phone_sensing']
        gps_weather_df = sensing_dct['gps_weather']

        phone_df[date_col_phone] = pd.to_datetime(phone_df[date_col_phone], format="mixed", errors="coerce")
        gps_weather_df[date_col_gps_weather] = pd.to_datetime(gps_weather_df[date_col_gps_weather], format="mixed", errors="coerce")

        # Merge the two DataFrames on the ID and day columns
        merged_df = pd.merge(
            phone_df,
            gps_weather_df,
            left_on=[self.raw_sensing_id_col, date_col_phone],  # columns from phone DataFrame
            right_on=[self.raw_sensing_id_col, date_col_gps_weather],  # columns from GPS/weather DataFrame
            how='outer'  # TODO which join makes sense here?
        )

        return merged_df

    def change_datetime_to_minutes(self, df: pd.DataFrame, **kwargs: str) -> pd.DataFrame:
        """
        This method changes the given columns that are in a datetime format (altough they still may be of dtype object)
        to minutes, so that the variables can be used by the ML models

        Args:
            df:
            *cols: columns that contain date values

        Returns:
            pd.DataFrame
        """

        for var_name in kwargs.values():
            col_name = self.config_parser(self.fix_cfg["sensing_based"]["gps_weather"],
                                          "continuous",
                                          var_name)[0]["item_names"][self.dataset]
            # Convert to datetime if the column is not already in datetime format
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            # Calculate the number of minutes since the start of the day (midnight)
            df[col_name] = df[col_name].dt.hour * 60 + df[col_name].dt.minute

        return df

    def apply_cut_offs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method filters the sensing variables and excludes values that exceed a certain threshold we define.
        These values are set to np.nan. The variable-specific threshold are stored in the config.

        Args:
            df:

        Returns:
            pd.DataFrame
        """
        return df  # TODO Implement

    def merge_state_df_sensing(self, df: pd.DataFrame, df_sensing: pd.DataFrame) -> pd.DataFrame:
        """
        This methods merges the sensing data to the trait-level data
        Using an outer join makes sense here
            - for the analysis without the sensing data, we use all samples that have questionnaire data
            - for the analysis with the sensing data, we may exclude this samples
            - furthermore, it could be possible that people do not have trait data, but esm and sensing data

        Args:
            df_states:
            df_sensing:

        Returns:
            pd.DataFrame
        """
        df = pd.merge(left=df,
                      right=df_sensing,
                      left_on=[self.raw_esm_id_col],
                      right_on=[self.raw_sensing_id_col],
                      how="left")
        return df
