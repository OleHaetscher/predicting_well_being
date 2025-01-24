import json
import os

import pandas as pd

from src.postprocessing.CVResultProcessor import CVResultProcessor
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.postprocessing.LinearRegressor import LinearRegressor
from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.SuppFileCreator import SuppFileCreator
from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.Logger import Logger
from src.utils.SanityChecker import SanityChecker
from src.utils.utilfuncs import merge_M_SD_in_dct, NestedDict


class Postprocessor:  # TODO adjust class / init doc at the end
    """
    Executes postprocessing steps for analyzing machine learning results and generating insights.

    Responsibilities include:
    - Conducting tests of significance to compare prediction results.
    - Calculating and displaying descriptive statistics.
    - Generating plots for results, SHAP values, SHAP interaction values and pred vs. true parity.
    - Preparing supplemental files for reports

    Attributes:
        cfg_preprocessing (NestedDict): Yaml config specifying details on preprocessing (e.g., scales, items).
        cfg_analysis (NestedDict): Yaml config specifying details on the ML analysis (e.g., CV, models).
        cfg_postprocessing (NestedDict): Yaml config specifying details on postprocessing (e.g., tables, plots).
        name_mapping (NestedDict): Mapping of feature names for presentation purposes.
        data_base_path: Path to the preprocessed data directory.
        base_result_path: Path to the base directory for storing results.
        cv_shap_results_path: Path to the main result directory containing cross-validation results and SHAP values.
        methods_to_apply: List of postprocessing methods to apply.
        datasets: List of datasets included in the analysis.
        meta_vars: List of meta variables to exclude from SHAP processing.
        data_loader: Instance of `DataLoader` for loading data.
        data_saver: Instance of `DataSaver` for saving data.
        raw_results_filenames: Filenames for raw analysis output.
        processed_results_filenames: Filenames for processed postprocessing results.
        full_data: The full dataset loaded from preprocessed data.
        logger: Instance of `Logger` for logging.
        descriptives_creator: Instance of `DescriptiveStatistics` for creating descriptive statistics.
        cv_result_processor: Instance of `CVResultProcessor` for processing cross-validation results.
        significance_testing: Instance of `SignificanceTesting` for conducting statistical tests.
        shap_processor: Instance of `ShapProcessor` for processing SHAP values.
        plotter: Instance of `ResultPlotter` for generating plots.
        supp_file_creator: Instance of `SuppFileCreator` for creating supplemental files.
        sanity_checker: Instance of `SanityChecker` for validating preprocessing and results.
    """

    def __init__(
        self,
        cfg_preprocessing: NestedDict,
        cfg_analysis: NestedDict,
        cfg_postprocessing: NestedDict,
        name_mapping: NestedDict,
    ):
        """
        Initializes the Postprocessor with configuration settings, paths, and analysis components.

        Args:
            cfg_preprocessing: Configuration dictionary for preprocessing settings,
                               including paths to data and logs.
            cfg_analysis: Configuration dictionary for analysis settings, such as cross-validation
                          and imputation parameters.
            cfg_postprocessing: Configuration dictionary for postprocessing settings,
                                including result paths, methods to apply, and filenames.
            name_mapping: Dictionary mapping feature combination keys to human-readable names.
        """
        self.cfg_preprocessing = cfg_preprocessing
        self.cfg_analysis = cfg_analysis
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.data_base_path = self.cfg_preprocessing["general"][
            "path_to_preprocessed_data"
        ]

        result_paths_cfg = self.cfg_postprocessing["general"]["data_paths"]
        self.base_result_path = result_paths_cfg["base_path"]
        self.cv_shap_results_path = os.path.join(
            self.base_result_path, result_paths_cfg["main_results"]
        )

        self.methods_to_apply = self.cfg_postprocessing["methods_to_apply"]
        self.datasets = self.cfg_preprocessing["general"]["datasets_to_be_included"]
        self.meta_vars = [
            self.cfg_analysis["cv"]["id_grouping_col"],
            self.cfg_analysis["imputation"]["country_grouping_col"],
            self.cfg_analysis["imputation"]["years_col"],
        ]

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()
        self.raw_results_filenames = self.cfg_analysis["output_filenames"]
        self.processed_results_filenames = self.cfg_postprocessing["general"][
            "processed_filenames"
        ]

        self.full_data = self.data_loader.read_pkl(
            os.path.join(
                self.data_base_path,
                self.cfg_preprocessing["general"]["full_data_filename"],
            )
        )

        self.logger = Logger(
            log_dir=self.cfg_preprocessing["general"]["log_dir"],
            log_file=self.cfg_preprocessing["general"]["log_name"],
        )

        self.descriptives_creator = DescriptiveStatistics(
            cfg_preprocessing=self.cfg_preprocessing,
            cfg_analysis=self.cfg_analysis,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=name_mapping,
            base_result_path=self.base_result_path,
            full_data=self.full_data,
        )

        self.cv_result_processor = CVResultProcessor(
            cfg_postprocessing=self.cfg_postprocessing,
        )

        self.significance_testing = SignificanceTesting(
            base_result_path=self.base_result_path,
            cfg_postprocessing=self.cfg_postprocessing,
        )

        self.shap_processor = ShapProcessor(
            cfg_postprocessing=self.cfg_postprocessing,
            base_result_path=self.base_result_path,
            cv_shap_results_path=self.cv_shap_results_path,
            processed_results_filenames=self.processed_results_filenames,
            name_mapping=self.name_mapping,
            meta_vars=self.meta_vars,
        )

        self.plotter = ResultPlotter(
            cfg_postprocessing=self.cfg_postprocessing,
            base_result_path=self.base_result_path,
        )

        self.supp_file_creator = SuppFileCreator(
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=self.name_mapping,
            meta_vars=self.meta_vars,
        )

        self.sanity_checker = SanityChecker(
            logger=self.logger,
            cfg_preprocessing=self.cfg_preprocessing,
            cfg_postprocessing=self.cfg_postprocessing,
            plotter=self.plotter,
        )

    def apply_methods(self) -> None:
        """
        Executes the methods specified in the cfg dynamically.

        Raises:
            ValueError: If a specified method does not exist in the class.
        """
        for method_name in self.cfg_postprocessing["methods_to_apply"]:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' is not implemented.")

            print(f">>>Executing postprocessing method: {method_name}<<<")
            getattr(self, method_name)()

    def condense_cv_results(self) -> None:
        """Summarizes the CV results for all analysis and stores the results in tables."""

        cv_results_filename = self.cfg_postprocessing["general"]["processed_filenames"][
            "cv_results_summarized"
        ]

        cfg_condense_results = self.cfg_postprocessing["condense_cv_results"]
        store_all_results = cfg_condense_results["all_results"]["store"]
        all_results_filename = cfg_condense_results["all_results"]["filename"]

        cv_results_dct = self.cv_result_processor.extract_cv_results(
            base_dir=self.cv_shap_results_path,
            metrics=cfg_condense_results["metrics"],
            cv_results_filename=cv_results_filename,
            negate_mse=cfg_condense_results["negate_mse"],
            decimals=cfg_condense_results["decimals"],
        )
        if store_all_results:
            all_result_path = os.path.join(
                self.cv_shap_results_path, all_results_filename
            )
            self.data_saver.save_json(cv_results_dct, all_result_path)

        for crit, crit_vals in cv_results_dct.items():
            for samples_to_include, samples_to_include_vals in crit_vals.items():
                for nnse_analysis in [True, False]:
                    if nnse_analysis and samples_to_include == "control":
                        continue

                    data_for_table = merge_M_SD_in_dct(samples_to_include_vals)

                    self.cv_result_processor.create_cv_results_table(
                        crit=crit,
                        samples_to_include=samples_to_include,
                        data=data_for_table,
                        output_dir=self.cv_shap_results_path,
                        nnse_analysis=nnse_analysis,
                        include_empty_col_between_models=True,
                    )

    def sanity_check_pred_vs_true(self) -> None:
        """Sanity checks the predictions vs. the true values for selected analysis and plots the results."""
        self.sanity_checker.sanity_check_pred_vs_true()

    def create_descriptives(self) -> None:
        """Creates tables containing descriptives (e.g., M, SD, correlations, reliability) for the datasets."""
        desc_cfg = self.cfg_postprocessing["create_descriptives"]
        var_table_cfg = desc_cfg["m_sd_table"]

        self.descriptives_creator.create_m_sd_var_table(
            vars_to_include=var_table_cfg["vars_to_include"],
            binary_stats_to_calc=var_table_cfg["bin_agg_lst"],
            continuous_stats_to_calc=var_table_cfg["cont_agg_dct"],
            table_decimals=var_table_cfg["decimals"],
            store_table=var_table_cfg["store"],
            filename=var_table_cfg["filename"],
            store_index=var_table_cfg["store_index"],
        )

        rel_dct = {}

        for dataset in self.datasets:
            traits_base_filename = self.cfg_postprocessing["create_descriptives"][
                "traits_base_filename"
            ]
            path_to_trait_df = os.path.join(
                self.data_base_path, f"{traits_base_filename}_{dataset}"
            )
            trait_df = (
                self.data_loader.read_pkl(path_to_trait_df)
                if os.path.exists(path_to_trait_df)
                else None
            )

            states_base_filename = self.cfg_postprocessing["create_descriptives"][
                "states_base_filename"
            ]
            path_to_state_df = os.path.join(
                self.data_base_path, f"{states_base_filename}_{dataset}"
            )
            state_df = (
                self.data_loader.read_pkl(path_to_state_df)
                if os.path.exists(path_to_state_df)
                else None
            )
            esm_id_col = self.cfg_preprocessing["general"]["esm_id_col"][dataset]
            esm_tp_col = self.cfg_preprocessing["general"]["esm_timestamp_col"][dataset]

            rel_dct[dataset] = self.descriptives_creator.compute_rel(
                state_df=state_df,
                trait_df=trait_df,
                dataset=dataset,
                decimals=desc_cfg["rel"]["decimals"],
            )

            wb_items_dct = self.descriptives_creator.create_wb_items_stats_per_dataset(
                dataset=dataset,
                state_df=state_df,
                trait_df=trait_df,
                esm_id_col=esm_id_col,
                esm_tp_col=esm_tp_col,
            )

            self.descriptives_creator.create_wb_items_table(
                m_sd_df=wb_items_dct["m_sd"],
                decimals=desc_cfg["wb_items"]["decimals"],
                store=desc_cfg["wb_items"]["store"],
                base_filename=desc_cfg["wb_items"]["filename"],
                dataset=dataset,
                icc1=wb_items_dct["icc1"],
                bp_corr=wb_items_dct["bp_corr"],
                wp_corr=wb_items_dct["wp_corr"],
                trait_corr=wb_items_dct["trait_corr"],
            )

        if desc_cfg["rel"]["store"]:
            file_path = os.path.join(
                self.descriptives_creator.desc_results_base_path,
                desc_cfg["rel"]["filename"],
            )
            self.data_saver.save_json(rel_dct, file_path)

    def conduct_significance_tests(self) -> None:
        """Conducts significance tests to compare models and compare predictor classes."""

        self.significance_testing.significance_testing()

    def create_cv_results_plots(self) -> None:
        """Creates a bar plot summarizing CV results for the analyses specified."""

        all_results_file_path = os.path.join(
            self.cv_shap_results_path,
            self.cfg_postprocessing["condense_cv_results"]["all_results"]["filename"],
        )
        all_cv_results_dct = self.data_loader.read_json(all_results_file_path)

        if all_cv_results_dct:
            self.plotter.plot_cv_results_plots_wrapper(
                cv_results_dct=all_cv_results_dct,
                rel=None,
            )

    def create_shap_plots(self) -> None:
        """Creates SHAP beeswarm plots for all analyses specified in the cfg."""

        self.plotter.plot_shap_beeswarm_plots(
            prepare_shap_data_func=self.shap_processor.prepare_shap_data,
            prepare_shap_ia_data_func=self.shap_processor.prepare_shap_ia_data,
        )

    def calculate_exp_lin_models(self) -> None:
        """Calculates explanatory linear models with the x best features for selected analyses."""

        linear_regressor_cfg = self.cfg_postprocessing["calculate_exp_lin_models"]

        for feature_combination in self.cfg_postprocessing["general"][
            "feature_combinations"
        ]["name_mapping"]["main"].keys():
            for samples_to_include in linear_regressor_cfg["samples_to_include"]:
                for crit in linear_regressor_cfg["crits"]:
                    for model_for_features in linear_regressor_cfg[
                        "model_for_features"
                    ]:
                        linear_regressor = LinearRegressor(
                            cfg_preprocessing=self.cfg_preprocessing,
                            cfg_analysis=self.cfg_analysis,
                            cfg_postprocessing=self.cfg_postprocessing,
                            name_mapping=self.name_mapping,
                            cv_shap_results_path=self.cv_shap_results_path,
                            df=self.full_data.copy(),
                            feature_combination=feature_combination,
                            crit=crit,
                            samples_to_include=samples_to_include,
                            model_for_features=model_for_features,
                            meta_vars=self.meta_vars,
                        )

                        linear_regressor.get_regression_data()
                        lin_model = linear_regressor.compute_regression_models()

                        linear_regressor.create_coefficients_table(
                            model=lin_model,
                            feature_combination=feature_combination,
                            output_dir=self.cv_shap_results_path,
                        )

    def create_lin_model_coefs_supp(self) -> None:
        """Creates a new dir containing JSON files with the coefficients of the linear models for each analysis."""

        filename = self.processed_results_filenames["lin_model_coefs_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"]["lin_coefs_output_filename"],
        )

        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=self.cv_shap_results_path,
            file_name=filename,
            output_base_dir=output_dir,
        )

    def create_shap_values_supp(self) -> None:
        """Creates a new dir containing JSON files with the shap_values for each analysis."""

        filename = self.processed_results_filenames["shap_values_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"]["shap_output_filename"],
        )

        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=self.cv_shap_results_path,
            file_name=filename,
            output_base_dir=output_dir,
        )

    def create_shap_ia_values_supp(self) -> None:
        """Creates a new dir containing JSON files with the shap_ia_values for some selected analysis."""

        filename = self.processed_results_filenames["shap_ia_values_summarized"]
        output_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["create_supp_files"][
                "shap_ia_output_filename"
            ],
        )

        input_dir = os.path.join(
            self.base_result_path,
            self.cfg_postprocessing["general"]["data_paths"]["ia_values"],

        )
        self.supp_file_creator.create_mirrored_dir_with_files(
            base_dir=input_dir,
            file_name=filename,
            output_base_dir=output_dir,
        )
