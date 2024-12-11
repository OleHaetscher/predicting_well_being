from typing import Sequence


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

