import os
import sys
from copy import copy

import pandas as pd


class DataSelector:
    """
    This class is responsible for selecting the data based on the given combination of features and samples.
    It takes the full dataframe containing as the input and reduces the features and the samples according
    to the given feature_combination - crit - samples_to_include combination as specified in the config
    """

    def __init__(
        self,
        var_cfg,
        df,
        feature_combination,
        crit,
        samples_to_include,
    ):
        """
        Constructor method of the DataSelector class.

        Args:
            var_cfg: YAML config determining which data to select for a specific analysis
            df: DataFrame containg all data (all datasets + all features + meta columns)
            feature_combination: Combination of features to use for the analysis, e.g., "pl_srmc_mac"
            crit: Criterion to predict, e.g., "wb_state"
            samples_to_include: Samples to include in the analysis, e.g., "selected"
        """
        # Analysis specifics
        self.var_cfg = var_cfg
        self.feature_combination = feature_combination
        self.crit = crit
        self.samples_to_include = samples_to_include

        self.id_grouping_col = self.var_cfg["analysis"]["cv"]["id_grouping_col"]
        self.country_grouping_col = self.var_cfg["analysis"]["imputation"]["country_grouping_col"]
        self.years_col = self.var_cfg["analysis"]["imputation"]["years_col"]
        self.meta_vars = [self.id_grouping_col, self.country_grouping_col, self.years_col]

        # Data
        self.df = df
        self.X = None
        self.y = None
        self.rows_dropped_crit_na = None

    def select_samples(self):
        """
        This method selects the samples based on the given combination using the indices that correspond
        to the samples (e.g., cocoesm_1). It applies the following logic:
            - for the analysis "selected" and "control", only selected datasets are used to reduce NaNs
            - for the analysis "all", all datasets are used, independent of the features used for the analysis
            - For the supplementary analyses: If a dataset does not contain the criterion, it is excluded

        Some more specifics:
            - For the "sens" analysis where samples_to_include == selected, only samples with sensing data are included
            - Another analysis was added to fit different features on a certain data subset (feature_combination -> _control)
            - If defined in the config, we take a sample for test purposes

        At the end of the function, it sets the filtered df as the class attribute.
        """
        if self.samples_to_include in ["selected", "control"]:
            datasets_included = self.var_cfg["analysis"]["feature_sample_combinations"][
                self.feature_combination
            ]
            datasets_included_filtered = [
                dataset
                for dataset in datasets_included
                if dataset in self.var_cfg["analysis"]["crit_available"][self.crit]
            ]
            self.datasets_included = datasets_included_filtered
            df_filtered = self.df[
                self.df.index.to_series().apply(
                    lambda x: any(
                        x.startswith(sample) for sample in self.datasets_included
                    )
                )
            ]

            if (
                self.samples_to_include in ["selected", "control"]
                and "sens" in self.feature_combination
            ):
                sens_columns = [
                    col for col in self.df.columns if col.startswith("sens_")
                ]
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

            # New -> Add control analysis with reduced samples
            if "_control" in self.feature_combination:
                sens_columns = [
                    col for col in self.df.columns if col.startswith("sens_")
                ]
                df_filtered = df_filtered[df_filtered[sens_columns].notna().any(axis=1)]

        else:  # samples_tp_include == "all"
            datasets_included_filtered = [
                dataset
                for dataset in self.var_cfg["analysis"]["feature_sample_combinations"][
                    "all_in"
                ]
                if dataset in self.var_cfg["analysis"]["crit_available"][self.crit]
            ]
            self.datasets_included = datasets_included_filtered
            df_filtered = self.df[
                self.df.index.to_series().apply(
                    lambda x: any(
                        x.startswith(sample) for sample in self.datasets_included
                    )
                )
            ]

        # It may also be possible that some people have NaNs on the trait wb measures -> exclude
        crit_col = f"crit_{self.crit}"
        df_filtered_crit_na = df_filtered.dropna(subset=[crit_col])
        self.rows_dropped_crit_na = len(df_filtered) - len(df_filtered_crit_na)

        # Sample for testing, if defined in the config
        if self.var_cfg["analysis"]["tests"]["sample"]:
            sample_size = self.var_cfg["analysis"]["tests"]["sample_size"]
            df_filtered_crit_na = df_filtered_crit_na.sample(
                n=sample_size, random_state=self.var_cfg["analysis"]["random_state"]
            )

        self.df = df_filtered_crit_na

    def select_features(self):
        """
        This method filters the columns of self.df according to the specifics of a given analysis.

        The config defines which features to include for a given analysis. Features of different categories
        can be differentiated by their prefix (e.g., pl_ for person-level features, i.e., personality traits,
        sociodemographics, and political attitudes).
        Some meta-columns are always included, as we need them later in the pipeline (e.g., grouping_id_col

        for ensuring correct train-test-splits)

        """
        selected_columns = copy(self.meta_vars)
        if self.samples_to_include in ["all", "selected"]:
            feature_prefix_lst = self.feature_combination.split("_")

            if "all_in" in self.feature_combination:  # include all features
                feature_prefix_lst = ["pl", "srmc", "sens", "mac"]
                if self.samples_to_include == "selected":
                    #self.logger.log(
                    #    f"    WARNING: No selected analysis needed for {self.feature_combination}, stop computations"
                    #)
                    sys.exit(0)

        elif self.samples_to_include == "control":
            feature_prefix_lst = ["pl"]

            no_control_lst = self.var_cfg["analysis"]["no_control_lst"]
            if self.feature_combination in no_control_lst:
                #self.logger.log(
                #    f"    WARNING: No control analysis needed for {self.feature_combination}, stop computations"
                #)
                sys.exit(0)
        else:
            raise ValueError(
                f"Invalid value #{self.samples_to_include}# for attr samples_to_include"
            )

        if self.feature_combination == "all":  # include all features
            feature_prefix_lst = ["pl", "srmc", "sens", "mac"]

        # always include grouping id
        for feature_cat in feature_prefix_lst:
            for col in self.df.columns:
                if col.startswith(feature_cat):
                    selected_columns.append(col)

        # remove neuroticism facets and self-esteem for selected analysis
        if "nnse" in self.feature_combination:
            to_remove = [
                "pl_depression",
                "pl_anxiety",
                "pl_emotional_volatility",
                "pl_self_esteem",
            ]
            selected_columns = [col for col in selected_columns if col not in to_remove]

        X = self.df[selected_columns].copy()

        setattr(self, "X", X)
        return X

    def select_criterion(self):
        """
        This method loads the criterion (reactivities, either EB estimates of random slopes or OLS slopes)
        according to the specifications in the var_cfg.
        It gets the specific name of the file that contains the data, connect it to the feature path for the
        current analysis and sets the loaded features as a class attribute "y".
        """
        y = self.df[f"crit_{self.crit}"]
        assert len(self.X) == len(
            y
        ), f"Features and criterion differ in length, len(X) == {len(self.X)}, len(y) == {len(y)}"

        setattr(self, "y", y)
        return y

    def select_best_features(self, df: pd.DataFrame, root_path: str, model: str, num_features: int = 10):
        """
        This method gets the best features for a given analysis setting
        (feature_combination, crit, samples_to_include, model) from the specific path
        and filters the df accordingly

        Args:
            df: The DataFrame containing the data
            root_path: The root path to the feature selection results
            model: mode, e.g., "randomforestregressor"

        Returns:
            Filtered df including only the x best features (and no meta columns)
        """
        file_path = os.path.join(
            root_path,
            self.feature_combination,
            self.samples_to_include,
            self.crit,
            model,
            f"top_{num_features}_features.txt",
        )

        with open(file_path, 'r') as file:
            feature_lst = [line.strip() for line in file]

        df_filtered = df[feature_lst]
        return df_filtered