from collections import defaultdict
from typing import Sequence

import pandas as pd


def apply_name_mapping(features: list, name_mapping: dict, prefix: bool) -> list:
    """
    Maps a list of features to their descriptive names based on the second-level name in the provided mapping dictionary.

    Args:
        features (list): A list of feature strings in the format "first_level_second_level".
        name_mapping (dict):

    Returns:
        list: A list of mapped second-level feature names.
    """
    mapped_features = []
    for feature in features:
        try:
            # Remove the feature category prefix (e.g., pl or srmc)
            if prefix:
                first_level, _, second_level = feature.partition('_')
                second_level_mapped = name_mapping[first_level][second_level]
                mapped_features.append(second_level_mapped)
            else:  # if no prefix, we process the crit_dfs
                second_level_mapped = name_mapping["crit"][feature]
                mapped_features.append(second_level_mapped)
        except KeyError:  # if meta-cols are passed
            print(f"No mapping found for {feature} ")
            mapped_features.append(feature)

    return mapped_features


def format_df(df: pd.DataFrame, capitalize: bool = False, columns: list = None, decimals: int = 2) -> pd.DataFrame:
    """
    Formats specified numerical columns of a DataFrame to the given number of decimal places.
    If no columns are specified, all numerical columns are formatted.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        capitalize (bool): If true, we capitalize the first word of the column name
        columns (list, optional): List of column names to format. Defaults to None.
        decimals (int): Number of decimal places to round to. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame with the specified (or all numerical) columns rounded to the given number of decimal places.
    """
    if columns is None:
        # Select only numerical columns if no specific columns are provided
        columns = df.select_dtypes(include=['number']).columns.tolist()

    for column in columns:
        if column in df:
            df[column] = df[column].apply(lambda x: custom_round(x, decimals))

    if capitalize:
        df.columns = df.columns.str.capitalize()
        # Capitalize string content in each column
        for col in df.select_dtypes(include=['object', 'string']):  # Select only string columns
            df[col] = df[col].str.capitalize()

    return df


def custom_round(value, decimals, max_decimals=10):
    """
    Custom rounding method to round to the specified number of decimals.
    If the rounded result is zero, recursively increase the precision.

    Args:
        value (float): The value to round.
        decimals (int): Number of decimal places to round to.
        max_decimals (int): Maximum precision to prevent infinite recursion. Defaults to 10.

    Returns:
        float: Rounded value with adjusted precision for small numbers.
    """
    if pd.isna(value) or value == 0:  # Keep NaN or exact zero unchanged
        return value

    rounded = round(value, decimals)

    if decimals == max_decimals:
        print(f"Max decimals reached for {value}")
        return 0

    if rounded == 0 and decimals < max_decimals:  # If rounded result is zero, increase precision
        return custom_round(value, decimals + 1, max_decimals)

    return rounded


def defaultdict_to_dict(dct):
    """

    Args:
        dct:

    Returns:

    """
    if isinstance(dct, defaultdict):
        dct = {k: defaultdict_to_dict(v) for k, v in dct.items()}
    return dct

