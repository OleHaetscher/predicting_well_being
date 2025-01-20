import json
import os

import pandas as pd
import pyreadr
import numpy as np

from src.utils.utilfuncs import create_defaultdict, defaultdict_to_dict, NestedDict


class DataLoader:
    """
    A utility class to load datasets in various formats (CSV, RData, pickle, JSON).

    Furthermore, it loads the metrics and coefficients from the processed output directory structure
    for further processing

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
        csv_files = [
            file
            for file in files
            if file.endswith(".csv")
            and os.path.isfile(os.path.join(path_to_dataset, file))
        ]

        if not csv_files:
            raise FileNotFoundError(f"No CSV datasets found in {path_to_dataset}")

        df_dct = {
            file[:-4]: pd.read_csv(
                os.path.join(path_to_dataset, file), encoding="latin", nrows=self.nrows
            )
            for file in csv_files
        }

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
        r_files = [
            file
            for file in files
            if file.endswith((".RData", ".rds"))
            and os.path.isfile(os.path.join(path_to_dataset, file))
        ]

        if not r_files:
            raise FileNotFoundError(f"No R datasets found in {path_to_dataset}")

        df_dct = {}
        for r_file in r_files:
            full_path = os.path.join(path_to_dataset, r_file)
            result = pyreadr.read_r(full_path)

            df_dct[r_file[:-4]] = next(
                iter(result.values())
            )  # Assuming one DataFrame per R file

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
        with open(path_to_dataset, "r") as f:
            data = json.load(f)
        return data

    def extract_cv_results(
            self,
            base_dir: str,
            metrics: list[str],
            cv_results_filename: str,
            correct_mse: bool = True,
            decimal: int = 3,
    ) -> NestedDict:
        """
        Extracts required metrics from `proc_cv_results.json` files within a nested directory structure.

        This method traverses the directory tree starting from `base_dir`, locates JSON files matching
        the given filename, and extracts summary statistics (mean and standard deviation) for the specified metrics.

        - Adjusts for negative MSE values when `correct_mse` is True.
        - Organizes results into a nested dictionary based on criteria, samples, feature combinations, and models.

        Args:
            base_dir: Base directory to search for CV results.
            metrics: List of metric names to extract (e.g., "neg_mean_squared_error").
            cv_results_filename: Name of the JSON file to read in each directory.
            correct_mse: If True, converts negative MSE values to positive. Defaults to True.
            decimal: Number of decimal places to round the extracted metrics. Defaults to 3.

        Returns:
            dict: A nested dictionary containing extracted metrics with the structure:
                  {crit: {samples_to_include: {feature_combination: {model: {metric: {"M": float, "SD": float}}}}}}.
        """
        result_dct = create_defaultdict(n_nesting=5, default_factory=dict)

        for root, _, files in os.walk(base_dir):
            if cv_results_filename in files:

                rearranged_key, crit, samples_to_include, feature_combination, model\
                    = self.rearrange_path_parts(root, base_dir, min_depth=4)

                try:
                    cv_results_summary = self.read_json(os.path.join(root, cv_results_filename))

                    for metric in metrics:
                        m_metric = cv_results_summary['m'][metric]
                        sd_metric = cv_results_summary['sd_across_folds_imps'][metric]

                        # Correct MSE values if required
                        if correct_mse and metric == "neg_mean_squared_error":
                            m_metric *= -1

                        rounded_m = f"{np.round(m_metric, 3):.3f}"
                        rounded_sd = f"{np.round(sd_metric, 3):.3f}"

                        result_dct[crit][samples_to_include][feature_combination][model][metric] = {
                            "M": rounded_m, "SD": rounded_sd
                        }
                except Exception as e:
                    print(f"Error reading {os.path.join(root, cv_results_filename)}: {e}")

        return defaultdict_to_dict(result_dct)

    @staticmethod
    def rearrange_path_parts(root, base_dir, min_depth=4):
        """
        Rearranges parts of a relative path if it meets the minimum depth requirement.

        Args:
            root (str): The full path to process.
            base_dir (str): The base directory to calculate the relative path from.
            min_depth (int): Minimum depth of the path to proceed.

        Returns:
            str or None: Rearranged path key if the depth requirement is met, else None.
        """
        relative_path = os.path.relpath(root, base_dir)
        path_parts = relative_path.strip(os.sep).split(os.sep)

        if len(path_parts) >= min_depth:
            # Do this as most appropriate for the tables
            feature_combination = path_parts[0]
            samples_to_include = path_parts[1]
            crit = path_parts[2]
            model = path_parts[3]
            rearranged_path_parts_joined = '_'.join([crit, samples_to_include, feature_combination, path_parts[1]])
            return (
                rearranged_path_parts_joined,
                crit,
                samples_to_include,
                feature_combination,
                model
            )

        else:
            print(f"Skipping directory {root} due to insufficient path depth.")
            return None


