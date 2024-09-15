from typing import Union

import pandas as pd
import numpy as np
import pytest

# Dummy DataChecker class with required methods
class DataChecker:
    def _map_column_to_binary(self, df: pd.DataFrame, column: str, mapping: dict) -> pd.Series:
        if df[column].dtype == object:
            return df[column].apply(lambda x: self._map_comma_separated(x, mapping))
        return df[column].map(lambda x: self._map_single_value(x, mapping))

    def _map_comma_separated(self, cell_value: str, mapping: dict, map_ambiguous: bool = True) -> Union[int, float]:
        if pd.isna(cell_value):
            return np.nan

        values = cell_value.split(',')
        mapped_values = [self._map_single_value(int(val.strip()), mapping) for val in values if val.strip().isdigit()]

        if not map_ambiguous:
            if len(set(mapped_values)) > 1:
                mapped_values = [np.nan]
        return max(mapped_values)

    @staticmethod
    def _map_single_value(value, mapping: dict) -> Union[int, float]:
        if pd.isna(value):
            return np.nan
        return mapping.get(value, 0)  # If not in mapping, return 0


# Tests for DataChecker
@pytest.fixture
def df():
    # Sample dataframe for testing
    return pd.DataFrame({
        'emotions': [1, 2, 1, np.nan, 0],
        'zpid': [1, 2, 3, np.nan, 0],
        'cocout': ['1', '2,3', '3', np.nan, ''],
        'pia': [1, 2, 3, np.nan, 0],
        'cocoesm': [1, 2, 3, np.nan, 0],
        'cocoms': [1, 2, 3, np.nan, 0],
    })

@pytest.fixture
def checker():
    # Create an instance of the DataChecker class
    return DataChecker()


def test_map_single_value(checker):
    # Test individual values using a simple mapping
    mapping = {1: 1}  # Only map 1 to 1, everything else should be 0 or NaN

    # Testing _map_single_value
    assert checker._map_single_value(1, mapping) == 1
    assert checker._map_single_value(2, mapping) == 0
    assert checker._map_single_value(3, mapping) == 0
    assert np.isnan(checker._map_single_value(np.nan, mapping))  # should return NaN
    assert checker._map_single_value(0, mapping) == 0


def test_map_comma_separated(checker):
    # Test mapping of comma-separated values
    mapping = {1: 1, 2: 1}  # Map 1 and 2 to 1, others to 0

    # Comma-separated values
    assert checker._map_comma_separated("1,2", mapping) == 1  # max of [1, 1]
    assert checker._map_comma_separated("1,3", mapping) == 1  # max of [1, 0]
    assert checker._map_comma_separated("3", mapping) == 0  # only 0 is mapped
    assert np.isnan(checker._map_comma_separated(np.nan, mapping))  # NaN input remains NaN


def test_map_column_to_binary(checker, df):
    # Test mapping on the entire dataframe column
    mappings = {
        'emotions': {1: 1},  # Only map '1' to 1 for emotions
        'zpid': {2: 1},  # Map '2' to 1 for zpid
        'cocout': {2: 1},  # Map '2' to 1 for cocout, check comma-separated
        'pia': {1: 1},  # Map '1' to 1 for pia
        'cocoesm': {1: 1},  # Map '1' to 1 for cocoesm
        'cocoms': {1: 1},  # Map '1' to 1 for cocoms
    }

    # Apply mapping for each column and check results
    mapped_emotions = checker._map_column_to_binary(df, 'emotions', mappings['emotions'])
    # Use pd.testing.assert_series_equal to compare, allowing for NaNs
    expected_emotions = pd.Series([1, 0, 1, np.nan, 0])
    pd.testing.assert_series_equal(mapped_emotions, expected_emotions, check_dtype=False, check_exact=False, check_names=False)

    mapped_zpid = checker._map_column_to_binary(df, 'zpid', mappings['zpid'])
    # Use pd.testing.assert_series_equal to compare, allowing for NaNs
    expected_zpid = pd.Series([0, 1, 0, np.nan, 0])
    pd.testing.assert_series_equal(mapped_zpid, expected_zpid, check_dtype=False, check_exact=False, check_names=False)
