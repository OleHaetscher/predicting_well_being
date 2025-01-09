from collections import defaultdict
from typing import Union, Any

import pandas as pd

# Define nested typing aliases for nested dictionaries
NestedDict = dict[Any, dict]


def apply_name_mapping(features: list[str], name_mapping: NestedDict, prefix: bool) -> list[str]:
    """
    Maps a list of features to their descriptive names based on the second-level name
    in the provided mapping dictionary.

    Args:
        features: A list of feature strings in the format "first_level_second_level".
        name_mapping: A dictionary containing the mappings for descriptive names.
                      The keys are first-level feature names, and the values are dictionaries
                      mapping second-level feature names to their descriptive names.
        prefix: If True, the feature strings include a prefix that should be used for mapping.

    Returns:
        list: A list of mapped second-level feature names or the original features if no mapping is found.
    """
    mapped_features = []
    for feature in features:
        try:
            if prefix:
                first_level, _, second_level = feature.partition('_')
                second_level_mapped = name_mapping[first_level][second_level]
                mapped_features.append(second_level_mapped)

            else:
                second_level_mapped = name_mapping["crit"][feature]
                mapped_features.append(second_level_mapped)

        except KeyError:  # if meta-cols are passed
            print(f"No mapping found for {feature} ")
            mapped_features.append(feature)

    return mapped_features


def format_df(
    df: pd.DataFrame,
    capitalize: bool = False,
    columns: list[str] = None,
    decimals: int = 2
) -> pd.DataFrame:
    """
    Formats specified numerical columns of a DataFrame to the given number of decimal places.
    If no columns are specified, all numerical columns are formatted. Optionally, capitalizes
    column names and string values.

    Args:
        df: The DataFrame containing the data.
        capitalize: If True, capitalize the first word of each column name and string values.
        columns: List of column names to format. If None, all numerical columns are formatted.
        decimals: Number of decimal places to round to.

    Returns:
        pd.DataFrame: A DataFrame with the specified (or all numerical) columns rounded to the given number of decimal places.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    for column in columns:
        if column in df:
            df[column] = df[column].apply(lambda x: custom_round(x, decimals))

    if capitalize:
        df.columns = df.columns.str.capitalize()
        for col in df.select_dtypes(include=['object', 'string']):
            df[col] = df[col].str.capitalize()

    return df


def custom_round(value: float, decimals: int, max_decimals: int = 10) -> float:
    """
    Custom rounding method to round to the specified number of decimals.
    If the rounded result is zero, recursively increases the precision to avoid losing small values.

    Args:
        value: The value to round.
        decimals: Number of decimal places to round to.
        max_decimals: Maximum precision to prevent infinite recursion.

    Returns:
        float: Rounded value with adjusted precision for small numbers.
    """
    if pd.isna(value) or value == 0:
        return value

    rounded = round(value, decimals)

    if decimals == max_decimals:
        print(f"Max decimals reached for {value}")
        return 0

    if rounded == 0 and decimals < max_decimals:  # If rounded result is zero, increase precision
        return custom_round(value, decimals + 1, max_decimals)

    return rounded


def defaultdict_to_dict(dct: Union[defaultdict, dict, NestedDict]) -> Union[dict, NestedDict]:
    """
    Recursively converts a defaultdict into a standard Python dictionary.

    Args:
        dct: The input dictionary, which may be a defaultdict or a nested dictionary containing defaultdict objects.

    Returns:
        dict: A standard Python dictionary with no defaultdict objects.
    """
    if isinstance(dct, defaultdict):
        dct = {k: defaultdict_to_dict(v) for k, v in dct.items()}
    return dct

