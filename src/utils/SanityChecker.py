import json
import os
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.utils.ConfigParser import ConfigParser
from src.utils.DataSaver import DataSaver
from src.utils.Logger import Logger
from src.utils.utilfuncs import NestedDict


class SanityChecker:
    """
    A class for performing various data sanity checks on a pandas DataFrame.

    This class provides configurable checks to ensure data integrity before processing.
    It logs results of each check, identifying issues such as missing values, data type mismatches,
    scale inconsistencies, and feature count mismatches.

    Attributes:
        logger (Any): A logger instance for logging sanity check results.
        cfg_preprocessing (NestedDict): Configuration settings for preprocessing data.
        cfg_postprocessing (NestedDict): Configuration settings for postprocessing data.
        cfg_sanity_checks (NestedDict): Thresholds and parameters for sanity checks.
        config_parser_class (Callable): A callable class for parsing configuration files.
        apply_to_full_df (bool): Indicates whether to apply checks to the full DataFrame.
        plotter: (ResultPlotter): Class that creates plots.
    """

    def __init__(
        self,
        logger: Logger,
        cfg_preprocessing: NestedDict,
        cfg_postprocessing: NestedDict = None,
        config_parser_class: ConfigParser = None,
        apply_to_full_df: bool = None,
        plotter: Any = None,
    ):
        """
        Initializes the SanityChecker with configuration and logging settings.

        Args:
            logger: A logger instance for logging the results of sanity checks.
            cfg_preprocessing: Configuration settings for preprocessing data.
            cfg_postprocessing: Configuration settings for postprocessing data.
            config_parser_class: A callable class for parsing configuration files.
            apply_to_full_df: Boolean specifying whether to apply checks to the full DataFrame.
            plotter. Class that creates plots.
        """
        self.logger = logger
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_postprocessing = cfg_postprocessing
        self.cfg_sanity_checks = self.cfg_preprocessing["sanity_checks"]

        self.config_parser_class = config_parser_class
        self.apply_to_full_df = apply_to_full_df
        self.plotter = plotter

        self.data_saver = DataSaver()

    def run_sanity_checks(
        self,
        df: pd.DataFrame,
        dataset: Optional[str] = None,
        df_before_final_sel: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Runs all configured sanity checks on the provided DataFrame.

        Args:
            df: The DataFrame to check.
            dataset: A string identifying the dataset (used for intermediate checks) or None for the final dataset.
            df_before_final_sel: A DataFrame representing the state before the final dataset selection.
        """
        self.logger.log("-----------------------------------------------")
        self.logger.log("  Conduct sanity checks")

        sanity_checks = [
            (self.check_nans, {"df": None}),
            (self.check_country, {"df": None, "dataset": dataset}),
            (self.check_dtypes, {"df": None}),
            (self.check_num_rows, {"df": None, "dataset": dataset}),
            (self.calc_reliability, {"df": df_before_final_sel, "dataset": dataset}),
            (self.check_zero_variance, {"df": None}),
            (self.check_scale_endpoints, {"df": None, "dataset": dataset}),
            (self.calc_correlations, {"df": None}),
        ]

        for method, kwargs in sanity_checks:
            kwargs = {k: v if v is not None else df for k, v in kwargs.items()}
            self._log_and_execute(method, **kwargs)

    def _log_and_execute(self, method: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Logs the execution of a sanity check method and runs it.

        Args:
            method: The sanity check method to execute.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Any: The result of the executed sanity check method.
        """
        self.logger.log(".")
        self.logger.log(f"  Executing {method.__name__}")

        return method(*args, **kwargs)

    def check_nans(self, df: pd.DataFrame) -> None:
        """
        Logs the percentage of NaNs in each column and warns if it exceeds the configured threshold.

        Args:
            df: The DataFrame to check for NaN values.
        """
        nan_thresh = self.cfg_sanity_checks["nan_thresh"]

        nan_per_col_summary = df.isna().mean()
        for col, nan_ratio_col in nan_per_col_summary.items():
            if nan_ratio_col > nan_thresh:
                self.logger.log(
                    f"    WARNING: Column '{col}' has {np.round(nan_ratio_col * 100,3)}% NaNs."
                )

        self.logger.log(".")
        nan_row_count = 0
        nan_per_row_summary = df.isna().mean(axis=1)
        for row, nan_ratio_row in nan_per_row_summary.items():
            if nan_ratio_row > nan_thresh:
                nan_row_count += 1

        self.logger.log(
            f"    WARNING: {nan_row_count} rows have more then {nan_thresh*100}% NaNs."
        )

        self.logger.log(".")
        cats_to_check = ["pl_", "srmc_", "sens_"]
        columns_to_check = [
            col for col in df.columns if any(cat in col for cat in cats_to_check)
        ]

        zero_non_nan_count = (df[columns_to_check].notna().sum(axis=1) == 0).sum()
        self.logger.log(
            f"    WARNING: {zero_non_nan_count} rows have 0 non-NaN values in {cats_to_check}."
        )

    def check_country(self, df: pd.DataFrame, dataset: str) -> None:
        """
        Ensures the "country" column in the DataFrame has no missing values.

        This method:
        - Checks for missing values (NaNs) in the "country" column.
        - Attempts to infer and fill missing countries based on a predefined mapping if the dataset allows it.
        - Logs warnings if any missing values remain, or if the inference is not possible for specific datasets.

        Args:
            df: The DataFrame to check and update.
            dataset: The name of the dataset being processed. Used to determine the country mapping logic.
        """
        country_mapping = self.cfg_sanity_checks["fix_country_mappings"]

        zero_non_nan_count_country = df["other_country"].isna().sum()
        if zero_non_nan_count_country > 0:
            self.logger.log(
                f"    WARNING: {zero_non_nan_count_country} in country column"
            )

            if dataset != "cocoesm":  # here we can infer the country
                country = country_mapping[dataset]
                self.logger.log(f"    Infer country {country} in {dataset}")
                df["other_country"] = df["other_country"].fillna(country)

            else:
                self.logger.log(f"    WARNING: Cannot infer country in dataset cocoesm")

    def check_dtypes(self, df: pd.DataFrame) -> None:
        """
        Validates that all columns in the DataFrame have numeric data types.

        This method:
        - Identifies columns with non-numeric data types.
        - Attempts to convert problematic columns to numeric, replacing invalid values with NaN.
        - Logs the results of the conversion process, including any columns that failed to convert.

        Args:
            df: The DataFrame to validate and potentially update.
        """
        for col, dtype in df.dtypes.items():
            if not pd.api.types.is_numeric_dtype(dtype):
                self.logger.log(
                    f"    WARNING: Column '{col}' has non-numeric dtype: {dtype}, try to convert"
                )
                df[col] = df[col].replace("not_a_number", np.nan)

                try:
                    df[col] = df[col].apply(
                        lambda x: pd.to_numeric(x) if not isinstance(x, tuple) else x
                    )
                    self.logger.log(f"      Conversion successful for column '{col}'")

                except ValueError:
                    self.logger.log(
                        f"      WARNING: Conversion was not successful. ML Models may fail "
                    )

    def check_num_rows(self, df: pd.DataFrame, dataset: str) -> None:
        """
        Logs the number of rows in the DataFrame and provides additional insights based on the dataset.

        This method:
        - Logs the total number of rows in the DataFrame.
        - For specific datasets, logs additional metrics like the number of rows with sensing data
          and the percentage of rows containing such data.
        - For specific datasets, it logs the rows per wave (i.e., cocoms, cocout).

        Args:
            df: The DataFrame to analyze.
            dataset: The name of the dataset being processed.

        """
        self.logger.log(f"    Num rows of df for {dataset}: {len(df)}")

        sens_columns = [col for col in df.columns if col.startswith("sens_")]
        if dataset == "zpid":
            df_sensing_filtered = df[df[sens_columns].notna().any(axis=1)]
            self.logger.log(
                f"    Num rows of df for {dataset} with sensing data: {len(df_sensing_filtered)}"
            )
            self.logger.log(
                f"    Percentage of samples that contain sensing data: "
                f"{np.round(len(df_sensing_filtered) / len(df), 3)}"
            )

        if dataset in ["cocout", "cocoms"]:
            for wave in df["other_studyWave"].unique():
                df_tmp = df[df["other_studyWave"] == wave]
                self.logger.log(f"      Num rows of df for {wave}: {len(df_tmp)}")

                if dataset == "cocoms":
                    df_sensing_filtered = df_tmp[
                        df_tmp[sens_columns].notna().any(axis=1)
                    ]
                    self.logger.log(
                        f"    Num rows of df for {wave} with sensing data: {len(df_sensing_filtered)}"
                    )
                    self.logger.log(
                        f"    Percentage of samples of df for {wave} that contain sensing data: "
                        f"{np.round(len(df_sensing_filtered) / len(df_tmp), 3)}"
                    )

    def log_num_features_per_cat(self, df: pd.DataFrame, dataset: str) -> None:
        """
        Logs the number of features present in each feature category within the DataFrame.

        This method:
        - Categorizes features based on predefined categories.
        - Logs the total count of features and the names of individual features for each category.

        Args:
            df: The DataFrame to analyze.
            dataset: The name of the dataset being processed.
        """
        # features_cats = ["pl", "srmc", "crit", "sens", "other"]
        features_cats = list(self.cfg_preprocessing["var_assignments"].keys())

        if dataset == "cocoesm":
            features_cats.append("mac")

        if dataset in ["coco_ms", "zpid"]:
            features_cats.append("sens")

        for cat in features_cats:
            self.logger.log(".")
            col_lst_per_cat = [
                f"{cat}_{col}" for col in self.cfg_preprocessing["var_assignments"][cat]
            ]
            cols_in_df = [col for col in df.columns if col in col_lst_per_cat]

            self.logger.log(
                f"        Number of columns for feature category {cat}: {len(cols_in_df)}"
            )
            for i in cols_in_df:
                self.logger.log(f"          {i}")

    def check_scale_endpoints(self, df: pd.DataFrame, dataset: str) -> None:
        """
        Validates that all column values lie within the expected scale endpoints.

        This method:
        - Iterates through the variable configuration to find expected scale endpoints.
        - Checks each column in the DataFrame for values outside the specified range.
        - Logs warnings for any out-of-bound values and raises an assertion error if issues are found.

        Args:
            df: The DataFrame to validate.
            dataset: The name of the dataset being processed.

        Raises:
            AssertionError: If any column has values outside the defined scale endpoints.
        """

        def get_column_names(cat: str, var_name: str) -> list[str]:
            """
            Resolves column names based on the variable category and variable name.

            Note: Implementation is a bit ugly but works correct and suffice for now.

            Args:
                cat: The category of the variable (e.g., "personality", "sociodemographics").
                var_name: The name of the variable.

            Returns:
                list[str]: A list of column names corresponding to the variable.
            """
            if cat in ["personality", "sociodemographics"]:
                return [f"pl_{var_name}"]

            elif cat == "self_reported_micro_context":
                if var_name in ["sleep_quality", "number_interaction_partners"]:
                    return [
                        f"srmc_{var_name}_mean",
                        f"srmc_{var_name}_sd",
                        f"srmc_{var_name}_min",
                        f"srmc_{var_name}_max",
                    ]
                else:
                    return [f"srmc_{var_name}"]

            elif cat == "criterion":
                return [f"crit_{var_name}"]

            else:
                return []

        errors = []

        for meta_cat in ["person_level", "esm_based"]:
            for cat, cat_entries in self.cfg_preprocessing[meta_cat].items():
                for var in cat_entries:
                    if "scale_endpoints" not in var or dataset not in var.get(
                        "item_names", []
                    ):
                        continue

                    if var["name"] in ["sleep_quality", "number_interaction_partners"]:
                        scale_min = 0
                    else:
                        scale_min = var["scale_endpoints"]["min"]
                    scale_max = var["scale_endpoints"]["max"]

                    column_names = get_column_names(cat, var["name"])

                    for col_name in column_names:
                        if col_name not in df.columns:
                            self.logger.log(
                                f"WARNING: Skip column '{col_name}', not found in DataFrame"
                            )
                            continue

                        column_values = df[col_name]
                        outside_values = column_values[
                            (column_values < scale_min) | (column_values > scale_max)
                        ]

                        if not outside_values.empty:
                            self.logger.log(
                                f"WARNING: Values out of scale bounds in column '{col_name}': {outside_values.tolist()}"
                            )
                            errors.append(
                                f"Column '{col_name}' contains values outside of scale endpoints {scale_min}-{scale_max}. "
                                f"Outliers: {outside_values.tolist()}"
                            )

        if errors:
            raise AssertionError("Scale validation failed:\n" + "\n".join(errors))

    def check_zero_variance(self, df: pd.DataFrame) -> None:
        """
        Identifies columns with zero variance in the DataFrame and logs these columns.

        This method:
        - Checks each column for zero variance, considering NaN values as equivalent to a single unique value.
        - Logs warnings for any columns with zero variance.

        Args:
            df: The DataFrame to check for zero variance.
        """
        zero_variance_columns = []

        for column in df.columns:
            unique_vals = df[column].dropna().unique()

            if len(unique_vals) == 1:
                zero_variance_columns.append(column)

        if zero_variance_columns:
            self.logger.log(f"    WARNING: Columns with zero variance")

            for i in zero_variance_columns:
                self.logger.log(f"          {i}")

    def calc_reliability(self, df: pd.DataFrame, dataset: str) -> None:
        """
        Computes and logs Cronbach's alpha for questionnaire scales in the DataFrame.

        This method:
        - Retrieves scale definitions from the configuration.
        - Skips scales with fewer than two items.
        - Logs a warning for scales with low Cronbach's alpha.

        Args:
            df: The DataFrame containing the questionnaire data.
            dataset: The name of the dataset being processed.
        """
        scale_entries = self.config_parser_class.find_key_in_config(
            cfg=self.cfg_preprocessing, key="scale_endpoints"
        )

        for scale in scale_entries:
            scale_name = scale["name"]
            if dataset in scale["item_names"]:
                item_names = scale["item_names"][dataset]

                if isinstance(item_names, str):
                    continue

                if isinstance(item_names, list):
                    if len(item_names) < 2:
                        continue

                items_df = df[item_names].dropna()

                # Calculate Cronbach's alpha
                alpha = pg.cronbach_alpha(data=items_df)[0]
                if np.isnan(alpha):
                    self.logger.log(
                        f"    WARNING: Not enough items to calculate Cronbach's alpha for {scale_name} in {dataset}."
                    )
                    continue

                if alpha < self.cfg_sanity_checks["cron_alpha_thresh"]:
                    self.logger.log(
                        f"    WARNING: Low reliability (alpha = {alpha:.3f}) for {scale_name} in {dataset}."
                    )
            else:
                continue

    def calc_correlations(self, df: pd.DataFrame) -> None:
        """
        Checks correlations between predefined columns (cfg) and logs any negative correlations.

        This method:
        - Extracts columns expected to have positive correlations.
        - Logs any correlations that are negative, as they might indicate suspicious data.

        Args:
            df: The DataFrame to analyze.
        """
        df_cols_adjusted = df.copy()
        df_cols_adjusted.columns = [
            col.split("_", 1)[1] if "_" in col else col for col in df.columns
        ]

        dataset_spec_cols = [
            col
            for col in df_cols_adjusted.columns
            if col in self.cfg_sanity_checks["expected_pos_corrs"]
        ]
        corr_table = df_cols_adjusted[dataset_spec_cols].corr()

        negative_correlations = []
        for i in range(len(corr_table.columns)):
            for j in range(i + 1, len(corr_table.columns)):  # Only check upper triangle
                corr_value = corr_table.iloc[i, j]
                if corr_value < 0:
                    negative_correlations.append(
                        (corr_table.columns[i], corr_table.columns[j], corr_value)
                    )

        if negative_correlations:
            for col1, col2, corr in negative_correlations:
                self.logger.log(
                    f"    WARNING: Negative correlation detected between '{col1}' and '{col2}': {corr:.3f}"
                )

    def sanity_check_sensing_data(
        self, df_sensing: pd.DataFrame, dataset: str
    ) -> pd.DataFrame:
        """
        Performs sanity checks on sensing data before computing summary statistics.

        This method:
        - Checks for a high percentage of NaN or zero values in sensing columns.
        - Logs descriptive statistics (mean, standard deviation, max) for each sensing variable.

        Args:
            df_sensing: The DataFrame containing sensing data.
            dataset: The name of the dataset being processed.

        Returns:
            pd.DataFrame: The updated DataFrame after performing the sanity checks.
        """
        vars_phone_sensing = self.config_parser_class.cfg_parser(
            self.cfg_preprocessing["sensing_based"]["phone"], "continuous"
        )
        vars_gps_weather = self.config_parser_class.cfg_parser(
            self.cfg_preprocessing["sensing_based"]["gps_weather"], "continuous"
        )
        total_vars = vars_phone_sensing + vars_gps_weather

        for sens_var in total_vars:
            col = sens_var["item_names"][dataset]
            nan_percentage = df_sensing[col].isna().sum() / len(df_sensing)

            if nan_percentage > self.cfg_sanity_checks["sensing"]["nan_col_thresh"]:
                self.logger.log(
                    f"        WARNING: {np.round(nan_percentage*100, 1)}% NaNs in {col}"
                )
            zero_percentage = (df_sensing[col] == 0).mean()

            if zero_percentage > self.cfg_sanity_checks["sensing"]["zero_col_thresh"]:
                self.logger.log(
                    f"        WARNING: {np.round(zero_percentage*100, 1)}% 0s in {col}"
                )

            mean = np.round(df_sensing[col].mean(), 3)
            sd = np.round(df_sensing[col].std(), 3)
            max_ = np.round(df_sensing[col].max(), 3)

            self.logger.log(f"    {sens_var['name']} M: {mean}, SD: {sd}, Max: {max_}")

        return df_sensing

    def sanity_check_pred_vs_true(self) -> None:
        """
        Analyzes predicted vs. true criterion values across samples to identify unexpected predictive patterns.

        This method:
        - Aggregates predicted and true values across repetitions, outer folds, and imputations.
        - Computes summary statistics and metrics (e.g., R², Spearman's rho) for each sample.
        - Generates parity plots to visualize predicted vs. true values.
        - Saves summary statistics and optional plots to the appropriate output directories.

        Configurable via:
            - Repetitions to check (`reps_to_check`).
            - Output paths and filenames (`data_paths`, `plot`, `summary_stats`).
        """
        if self.cfg_postprocessing:
            cfg_pred_vs_true = self.cfg_postprocessing["sanity_check_pred_vs_true"]
            root_dir = os.path.join(
                self.cfg_postprocessing["general"]["data_paths"]["base_path"],
                self.cfg_postprocessing["general"]["data_paths"]["pred_vs_true"],
            )
            reps_to_check = cfg_pred_vs_true["reps_to_check"]
            stats_decimals = cfg_pred_vs_true["summary_stats"]["decimals"]

            for dirpath, _, filenames in os.walk(root_dir):
                if not filenames:
                    continue

                index_data = {}
                for filename in filenames:
                    if filename.startswith("pred_vs_true_rep_") and filename.endswith(
                        ".json"
                    ):
                        rep_number = filename.removeprefix(
                            "pred_vs_true_rep_"
                        ).removesuffix(".json")
                        if not (
                            rep_number.isdigit() and int(rep_number) in reps_to_check
                        ):
                            continue

                        file_path = os.path.join(dirpath, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)

                        for outer_fold_data in data.values():
                            for imp_data in outer_fold_data.values():
                                for index, pred_true in imp_data.items():
                                    index_data.setdefault(index, []).append(pred_true)

                if not index_data:
                    continue

                # Process sample data
                sample_data = {}
                for index, pred_true_list in index_data.items():
                    sample_name = index.split("_")[0]
                    sample_data.setdefault(
                        sample_name, {"pred": [], "true": [], "diff": []}
                    )

                    for pred, true in pred_true_list:
                        sample_data[sample_name]["pred"].append(pred)
                        sample_data[sample_name]["true"].append(true)
                        sample_data[sample_name]["diff"].append(true - pred)

                # Generate parity plot
                dir_components = os.path.normpath(dirpath).split(os.sep)
                self.plotter.plot_pred_true_parity(
                    sample_data=sample_data,
                    samples_to_include=dir_components[-3],
                    crit=dir_components[-2],
                    model=dir_components[-1],
                    store_plot=cfg_pred_vs_true["plot"]["store"],
                    filename=cfg_pred_vs_true["plot"]["filename"],
                )

                # Compute summary statistics
                summary_statistics = {
                    sample_name: {
                        "pred_mean": np.round(np.mean(values["pred"]), stats_decimals),
                        "pred_std": np.round(np.std(values["pred"]), stats_decimals),
                        "true_mean": np.round(np.mean(values["true"]), stats_decimals),
                        "true_std": np.round(np.std(values["true"]), stats_decimals),
                        "diff_mean": np.round(np.mean(values["diff"]), stats_decimals),
                        "diff_std": np.round(np.std(values["diff"]), stats_decimals),
                        "r2_score": np.round(
                            r2_score(values["true"], values["pred"]), stats_decimals
                        )
                        if len(values["pred"]) > 1
                        else None,
                        "spearman_rho": np.round(
                            spearmanr(values["true"], values["pred"])[0], stats_decimals
                        )
                        if len(values["pred"]) > 1
                        else None,
                    }
                    for sample_name, values in sample_data.items()
                }

                # Save summary statistics
                if cfg_pred_vs_true["summary_stats"]["store"]:
                    output_file = os.path.join(
                        dirpath, cfg_pred_vs_true["summary_stats"]["filename"]
                    )
                    self.data_saver.save_json(summary_statistics, output_file)
