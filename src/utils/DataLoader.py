import json
import os
import pandas as pd
import pyreadr

class DataLoader:
    def __init__(self, nrows: int = None):
        """
        Initializes the DataLoader with the specified number of rows to load (for csv files).

        Args:
            nrows (int): Number of rows to read from each file. For R files, it will load all rows.
        """
        self.nrows = nrows

    def read_csv(self, path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        This method loads all CSV files contained in the specified path and returns a dictionary of pandas DataFrames.

        Args:
            path_to_dataset (str): Path to the directory containing the CSV files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary where the keys are file names (without extensions)
                                     and the values are pandas DataFrames.

        Raises:
            FileNotFoundError: If no CSV files are found in the specified directory.
        """
        # List all files in the directory and filter for CSV files
        files = os.listdir(path_to_dataset)
        csv_files = [file for file in files if file.endswith('.csv') and os.path.isfile(os.path.join(path_to_dataset, file))]

        if not csv_files:
            raise FileNotFoundError(f"No CSV datasets found in {path_to_dataset}")

        # Read each CSV file into a DataFrame
        df_dct = {file[:-4]: pd.read_csv(os.path.join(path_to_dataset, file), encoding="latin", nrows=self.nrows)
                  for file in csv_files}

        # Clean up DataFrames by dropping unnecessary columns
        for key, df in df_dct.items():
            if "Unnamed: 0" in df.columns:
                df_dct[key] = df.drop(["Unnamed: 0"], axis=1)

        return df_dct

    def read_r(self, path_to_dataset: str) -> dict[str, pd.DataFrame]:
        """
        This method loads all R data files (.RData or .rds) contained in the specified path and returns a dictionary of pandas DataFrames.

        Args:
            path_to_dataset (str): Path to the directory containing the R files.

        Returns:
            dict[str, pd.DataFrame]: A dictionary where the keys are file names (without extensions)
                                     and the values are pandas DataFrames.

        Raises:
            FileNotFoundError: If no R files are found in the specified directory.
        """
        # List all files in the directory and filter for RData or rds files
        files = os.listdir(path_to_dataset)
        r_files = [file for file in files if file.endswith(('.RData', '.rds')) and os.path.isfile(os.path.join(path_to_dataset, file))]

        if not r_files:
            raise FileNotFoundError(f"No R datasets found in {path_to_dataset}")

        # Read each R file into a DataFrame
        df_dct = {}
        for r_file in r_files:
            full_path = os.path.join(path_to_dataset, r_file)
            result = pyreadr.read_r(full_path)  # Read RData or rds file, returns a dictionary of variables
            # Store the DataFrame(s) using the file name (without extension) as the key
            df_dct[r_file[:-4]] = next(iter(result.values()))  # Assuming one DataFrame per R file

        return df_dct

    def read_pkl(self, path_to_dataset: str) -> pd.DataFrame:
        """
        This method loads a pickle file from a given directory and returns a dataframe.

        Args:
            path_to_dataset:

        Returns:
            pd.DataFrame
        """
        df = pd.read_pickle(path_to_dataset)
        return df

    def read_json(self, path_to_dataset: str) -> pd.DataFrame:
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
