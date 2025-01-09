import json
import os

import pandas as pd
import pyreadr


class DataLoader:
    """
    A utility class to load datasets in various formats (CSV, RData, pickle, JSON).

    Attributes:
        nrows (int): Number of rows to read when loading CSV files. If None, all rows are loaded.
    """
    def __init__(self, nrows: int = None) -> None:
        """
        Initializes the DataLoader with the specified number of rows to load (for CSV files).

        Args:
            nrows: Number of rows to read (currently only implemented for CSV files).
        """
        self.nrows = nrows

    def read_csv(self, path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        Loads all CSV files in the specified directory and returns them as pandas DataFrames.

        - Each CSV file is loaded into a DataFrame.
        - Drops any unnecessary "Unnamed: 0" column from the DataFrames.

        Args:
            path_to_dataset: Path to the directory containing the CSV files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary with:
                - Keys: File names without extensions.
                - Values: Corresponding DataFrames.

        Raises:
            FileNotFoundError: If no CSV files are found in the specified directory.
        """
        files = os.listdir(path_to_dataset)
        csv_files = [file for file in files if file.endswith('.csv') and os.path.isfile(os.path.join(path_to_dataset, file))]

        if not csv_files:
            raise FileNotFoundError(f"No CSV datasets found in {path_to_dataset}")

        df_dct = {file[:-4]: pd.read_csv(os.path.join(path_to_dataset, file), encoding="latin", nrows=self.nrows)
                  for file in csv_files}

        for key, df in df_dct.items():
            if "Unnamed: 0" in df.columns:
                df_dct[key] = df.drop(["Unnamed: 0"], axis=1)

        return df_dct

    @staticmethod
    def read_r(path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        Loads all R data files (.RData or .rds) in the specified directory and returns them as pandas DataFrames.

        Args:
            path_to_dataset: Path to the directory containing the R files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary with:
                - Keys: File names without extensions.
                - Values: Corresponding DataFrames.

        Raises:
            FileNotFoundError: If no R files are found in the specified directory.
        """
        files = os.listdir(path_to_dataset)
        r_files = [file for file in files if file.endswith(('.RData', '.rds')) and os.path.isfile(os.path.join(path_to_dataset, file))]

        if not r_files:
            raise FileNotFoundError(f"No R datasets found in {path_to_dataset}")

        df_dct = {}
        for r_file in r_files:
            full_path = os.path.join(path_to_dataset, r_file)
            result = pyreadr.read_r(full_path)

            df_dct[r_file[:-4]] = next(iter(result.values()))  # Assuming one DataFrame per R file

        return df_dct

    @staticmethod
    def read_pkl(path_to_dataset: str) -> pd.DataFrame:
        """
        Loads a pickle file from the specified path and returns it as a DataFrame.

        Args:
            path_to_dataset: Path to the pickle file.

        Returns:
            pd.DataFrame: DataFrame loaded from the pickle file.
        """
        df = pd.read_pickle(path_to_dataset)
        return df

    @staticmethod
    def read_json(path_to_dataset: str) -> pd.DataFrame:
        """
        This method loads a JSON file from a given directory and returns a dataframe.

        Args:
            path_to_dataset (str): The path to the JSON file to be loaded.

        Returns:
            pd.DataFrame: DataFrame loaded from the JSON file.
        """
        with open(path_to_dataset, 'r') as f:
            data = json.load(f)
        return data
