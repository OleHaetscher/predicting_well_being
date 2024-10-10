from typing import Any, Callable

import numpy as np
import pandas as pd
import pingouin as pg


class SanityChecker:
    """
    A class that performs data sanity checks on a pandas DataFrame.

    Methods:
        check_nans: Checks the number of NaNs in each column and logs if there are too many.
        check_variance: Check if certain columns have no variance at all.
        check_dtypes: Logs if any column has non-numeric data types.
        calculate_statistics: Placeholder method to calculate scale reliability or correlations.
        check_scale_endpoints: Should correspond to the range given in the config, and should be the same across studies
        check number of columns == number of features in PreReg



    """

    def __init__(self, logger, fix_cfg, cfg_sanity_checks, config_parser_class, apply_to_full_df):  # more?
        """
        Initializes the DataChecker class with the dataframe and NaN threshold.

        Args:
            df (pd.DataFrame): The input dataframe to perform checks on.
            nan_threshold (float): The threshold of NaN percentage to log a warning.
        """
        self.logger = logger
        self.fix_cfg = fix_cfg
        self.cfg_sanity_checks = cfg_sanity_checks
        self.config_parser_class = config_parser_class
        self.apply_to_full_df = apply_to_full_df  # bool

    def run_sanity_checks(self, df: pd.DataFrame, dataset: str | None, df_before_final_sel: pd.DataFrame | None) -> None:
        """
        This function runs all sanity checks and log the results.

        Args:
            df:
            dataset: Specific str for dataset (if this is applied before the final dataset merge)
                or None, if it is applied to the final dataframe
            df_before_final_sel: The state df we need for computing scale reliabilities

        Returns:
            None
        """
        self.logger.log("-----------------------------------------------")
        self.logger.log("  Conduct sanity checks")
        sanity_checks = [
            (self.check_nans, {'df': None}),
            (self.check_dtypes, {'df': None}),
            (self.check_num_features, {'df': None, 'dataset': dataset}),
            (self.check_num_rows, {'df': None, 'dataset': dataset}),
            (self.calc_reliability, {'df': df_before_final_sel, 'dataset': dataset}),
            (self.check_zero_variance, {'df': None}),
            (self.check_scale_endpoints, {'df': None}),
            (self.calc_correlations, {'df': None}),
        ]
        for method, kwargs in sanity_checks:
            kwargs = {k: v if v is not None else df for k, v in kwargs.items()}
            self._log_and_execute(method, **kwargs)

    def _log_and_execute(self, method: Callable, *args: Any, **kwargs: Any):
        """

        Args:
            method:
            *args:
            **kwargs:

        Returns:

        """
        self.logger.log(".")
        self.logger.log(f"  Executing {method.__name__}")
        return method(*args, **kwargs)

    def check_nans(self, df: pd.DataFrame) -> None:
        """
        Logs the percentage of NaNs in each column and warns if it exceeds the threshold.
        """
        nan_thresh = self.cfg_sanity_checks["nan_thresh"]

        nan_per_col_summary = df.isna().mean()
        for col, nan_ratio_col in nan_per_col_summary.items():
            if nan_ratio_col > nan_thresh:
                self.logger.log(f"    WARNING: Column '{col}' has {np.round(nan_ratio_col,3) * 100}% NaNs.")

        self.logger.log(".")
        nan_per_row_summary = df.isna().mean(axis=1)
        for row, nan_ratio_row in nan_per_row_summary.items():
            if nan_ratio_row > nan_thresh:
                self.logger.log(f"    WARNING: Row '{row}' has {np.round(nan_ratio_row,3) * 100}% NaNs.")

    def check_dtypes(self, df: pd.DataFrame) -> None:
        """
        Requires that all columns must have numeric data types.

        Args:
            df:

        Returns:

        """
        for col, dtype in df.dtypes.items():
            if not pd.api.types.is_numeric_dtype(dtype):
                self.logger.log(f"    WARNING: Column '{col}' has non-numeric dtype: {dtype}, try to convert")
                df[col] = df[col].replace("not_a_number", np.nan)
                try:
                    df[col] = pd.to_numeric(df[col])
                    self.logger.log(f"      Conversion successful for column '{col}'")
                except ValueError:
                    self.logger.log(f"      WARNING: Conversion was not successful. ML Models may fail ")

    def check_num_features(self, df: pd.DataFrame, dataset: str) -> None:
        """
        This function evaluates if the number of features in the final preprocessed dataframe
        corresponds to the number of features in the PreReg (Table S1).
        If this does not hold,
        Args:
            df:
            dataset:

        Returns:
        """
        num_cols_expected = self.cfg_sanity_checks["number_of_features"][dataset]
        if len(df.columns) != num_cols_expected:
            self.logger.log(f"    WARNING: Number of columns in PreReg: {num_cols_expected} Number of columns in df: {len(df.columns)}")
        else:
            self.logger.log(f"    Number of columns in PreReg corresponds to number of columns in data")
        self.log_num_features_per_cat(df, dataset)  # do it either way to check unexpected things

    def check_num_rows(self, df: pd.DataFrame, dataset: str) -> None:
        """

        Args:
            df:
            dataset:

        Returns:

        """
        self.logger.log(f"    Num rows of df for {dataset}: {len(df)}")

        sens_columns = [col for col in df.columns if col.startswith("sens_")]
        if dataset == "zpid":
            df_sensing_filtered = df[df[sens_columns].notna().any(axis=1)]
            self.logger.log(f"    Num rows of df for {dataset} with sensing data: {len(df)}")
            self.logger.log(f"    Percentage of samples that contain sensing data: "
                            f"{np.round(len(df_sensing_filtered) / len(df), 3)}")

        if dataset in ["cocout", "cocoms"]:
            for wave in df["other_studyWave"].unique():
                df_tmp = df[df["other_studyWave"] == wave]
                self.logger.log(f"      Num rows of df for {wave}: {len(df_tmp)}")

                if dataset == "cocoms":
                    df_sensing_filtered = df_tmp[df_tmp[sens_columns].notna().any(axis=1)]
                    self.logger.log(f"    Num rows of df for {wave} with sensing data: {len(df_sensing_filtered)}")
                    self.logger.log(f"    Percentage of samples of df for {wave} that contain sensing data: "
                                    f"{np.round(len(df_sensing_filtered) / len(df_tmp), 3)}")

    def log_num_features_per_cat(self, df: pd.DataFrame, dataset: str) -> None:
        """
        This function is executed if there is a differences in the number of expected and present features.
        It further checks which feature category causes the difference.

        Args:
            df:
            dataset:

        Returns:
        """
        features_cats = ["pl", "srmc", "crit", "other"]
        if dataset == "cocoesm":
            features_cats.append("mac")
        if dataset in ["coco_ms", "zpid"]:
            features_cats.append("sens")
        for cat in features_cats:
            self.logger.log(".")
            col_lst_per_cat = [f"{cat}_{col}" for col in self.fix_cfg["var_assignments"][cat]]
            cols_in_df = [col for col in df.columns if col in col_lst_per_cat]
            self.logger.log(f"        Number of columns for feature category {cat}: {len(cols_in_df)}")
            for i in cols_in_df:
                self.logger.log(f"          {i}")

    def check_scale_endpoints(self, df: pd.DataFrame) -> None:
        """
        This method checks if the range of values in all cells corresponds to the scale endpoints defined in the cfg.
        If not, it logs all values per column that lie outside the defined range and throws and assertion error

        Args:
            df:

        Returns:

        """
        errors = []  # To collect any errors found

        for meta_cat in ["person_level", "esm_based"]:
            for cat, cat_entries in self.fix_cfg[meta_cat].items():
                for var in cat_entries:
                    if 'scale_endpoints' in var:
                        col_names_org = [col.split('_', 1)[1] if '_' in col else col for col in df.columns]
                        column_names = [i for i in df.columns if i in col_names_org]
                        scale_min = var['scale_endpoints']['min']
                        scale_max = var['scale_endpoints']['max']

                        for col in column_names:
                            # Get the column values
                            column_values = df[col]

                            # Check if any values are outside the min/max range
                            outside_values = column_values[(column_values < scale_min) | (column_values > scale_max)]

                            if not outside_values.empty:
                                # Log the offending values
                                self.logger.log(f"    WARNING: Values out of scale bounds in column '{col}': {outside_values.tolist()}")
                                errors.append(
                                    f"      Column '{col}' contains values outside of scale endpoints {scale_min}-{scale_max}. "
                                    f"      Outliers: {outside_values.tolist()}")

                # Raise an assertion error if any issues were found
                if errors:
                    raise AssertionError("Scale validation failed:\n" + "\n".join(errors))

    def check_zero_variance(self, df: pd.DataFrame) -> None:
        """
        This method checks if any column in the DataFrame has zero variance and logs these columns.
        Note: This should also include columns that contain np.nan and one other value

        Args:
            df (pd.DataFrame): The DataFrame to check for zero variance.

        Returns:
            None: Logs the columns with zero variance but doesn't return any value.
        """
        zero_variance_columns = []

        # Iterate through each column in the DataFrame
        for column in df.columns:
            # Check if the column has only one unique value or one unique value with NaNs
            unique_vals = df[column].dropna().unique()  # Exclude NaNs when checking unique values
            if len(unique_vals) == 1:  # Only one unique value remains (e.g., 0 or another constant)
                zero_variance_columns.append(column)

        if zero_variance_columns:
            self.logger.log(f"    WARNING: Columns with zero variance")
            for i in zero_variance_columns:
                self.logger.log(f"          {i}")

    def calc_reliability(self, df: pd.DataFrame, dataset: str) -> None:
        """
        This method calculates the reliability for each questionnaire scale. 1-item entires are skipped.
        It logs if the reliability (i.e., cronbach's alpha is below a certain threshold).

        Args:
            df (pd.DataFrame): The DataFrame containing the questionnaire data.

        Returns:
            None
        """
        # Get all scale entries with the 'scale_endpoints' key
        scale_entries = self.config_parser_class.find_key_in_config(cfg=self.fix_cfg, key="scale_endpoints")

        for scale in scale_entries:
            scale_name = scale['name']
            if dataset in scale["item_names"]:
                item_names = scale["item_names"][dataset]

                if isinstance(item_names, str):
                    continue

                if isinstance(item_names, list):
                    if len(item_names) < 2:
                        continue

                # Extract the relevant columns (items) from the DataFrame
                items_df = df[item_names].dropna()

                # Calculate Cronbach's alpha
                alpha = pg.cronbach_alpha(data=items_df)[0]
                if np.isnan(alpha):
                    self.logger.log(f"    WARNING: Not enough items to calculate Cronbach's alpha for {scale_name} in {dataset}.")
                    continue

                # Log only if Cronbach's alpha is below the threshold
                if alpha < self.cfg_sanity_checks["cron_alpha_thresh"]:
                    self.logger.log(f"    WARNING: Low reliability (alpha = {alpha:.3f}) for {scale_name} in {dataset}.")
            else:
                continue

    def calc_correlations(self, df: pd.DataFrame) -> None:
        """
        This method calculate a number of correlations that are theoretically expected to be positive.
        It logs the correlations that are negative (and thus, suspicious)

        Args:
            df:
        Returns:
            None
        """
        df_cols_adjusted = df.copy()
        df_cols_adjusted.columns = [col.split('_', 1)[1] if '_' in col else col for col in df.columns]
        dataset_spec_cols = [col for col in df_cols_adjusted.columns if col in self.cfg_sanity_checks["expected_pos_corrs"]]
        corr_table = df_cols_adjusted[dataset_spec_cols].corr()

        # Iterate through the correlation matrix and find negative correlations
        negative_correlations = []
        for i in range(len(corr_table.columns)):
            for j in range(i + 1, len(corr_table.columns)):  # Only check upper triangle
                corr_value = corr_table.iloc[i, j]
                if corr_value < 0:
                    negative_correlations.append((corr_table.columns[i], corr_table.columns[j], corr_value))

        # Log any negative correlations
        if negative_correlations:
            for col1, col2, corr in negative_correlations:
                self.logger.log(f"    WARNING: Negative correlation detected between '{col1}' and '{col2}': {corr:.3f}")

    def sanity_check_sensing_data(self, df_sensing: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """
        This method does some sanity checks for the sensing data. It makes more sens to do these checks before
        calculating the summary statistics, as some things would be unnoticed otherwise (e.g., large amount of
        0s or NaNs that may bias the statistics). Therefore, this function is applied before the general
        sanity checking before computing the summary statistics.

        Returns:

        """
        vars_phone_sensing = self.config_parser_class.cfg_parser(self.fix_cfg["sensing_based"]["phone"],
                                                                 "continuous")
        vars_gps_weather = self.config_parser_class.cfg_parser(self.fix_cfg["sensing_based"]["gps_weather"],
                                                               "continuous")
        total_vars = vars_phone_sensing + vars_gps_weather
        for sens_var in total_vars:
            col = sens_var["item_names"][dataset]
            nan_percentage = df_sensing[col].isna().sum() / len(df_sensing)
            if nan_percentage > self.cfg_sanity_checks["sensing"]["nan_col_thresh"]:
                self.logger.log(f"        WARNING: {np.round(nan_percentage*100, 1)}% NaNs in {col}")
            zero_percentage = (df_sensing[col] == 0).mean()
            if zero_percentage > self.cfg_sanity_checks["sensing"]["zero_col_thresh"]:
                self.logger.log(f"        WARNING: {np.round(zero_percentage*100, 1)}% 0s in {col}")

        return df_sensing
