import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.postprocessing.LinearRegressor import LinearRegressor
from src.postprocessing.ResultPlotter import ResultPlotter
from src.postprocessing.ResultTableCreator import ResultTableCreator
from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.utils.DataLoader import DataLoader
from src.utils.DataSaver import DataSaver
from src.utils.DataSelector import DataSelector
from src.utils.Logger import Logger
from src.utils.SanityChecker import SanityChecker
from src.utils.utilfuncs import custom_round, apply_name_mapping, defaultdict_to_dict, merge_M_SD_in_dct


class Postprocessor:
    """
    This class executes the different postprocessing steps. This includes
        - conducting tests of significance to compare prediction results
        - calculate and display descriptive statistics
        - creates plots (results, SHAP, SHAP interaction values)
    """
    # TODO: Adjust config stuff -> var_cfg and fix_cfg should be replaced
    def __init__(self, fix_cfg, var_cfg, cfg_postprocessing, name_mapping):
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]
        self.processed_output_path = self.var_cfg["postprocessing"]["processed_results_path"]
        self.data_base_path = self.var_cfg["preprocessing"]["path_to_preprocessed_data"]

        self.cv_results_dct = {}
        self.metrics = self.var_cfg["postprocessing"]["metrics"]
        self.methods_to_apply = self.var_cfg["postprocessing"]["methods"]
        self.datasets = self.var_cfg["general"]["datasets_to_be_included"]
        self.meta_vars = [
            self.var_cfg["analysis"]["cv"]["id_grouping_col"],
            self.var_cfg["analysis"]["imputation"]["country_grouping_col"],
            self.var_cfg["analysis"]["imputation"]["years_col"]
        ]
        self.n_samples_dct = None

        self.data_loader = DataLoader()
        self.data_saver = DataSaver()

        # Extract some variables that we need frequently in the more specialized classes



        self.logger = Logger(
            log_dir=self.var_cfg["general"]["log_dir"],
            log_file=self.var_cfg["general"]["log_name"]
        )

        self.full_data = self.data_loader.read_pkl(
            os.path.join(self.data_base_path, "full_data")
        )

        self.descriptives_creator = DescriptiveStatistics(
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=name_mapping,
            full_data=self.full_data,
        )
        self.result_table_creator = ResultTableCreator(
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=name_mapping
        )

        self.significance_testing = SignificanceTesting(
            base_result_dir=self.processed_output_path,
            cfg_postprocessing=self.cfg_postprocessing,
        )
        self.shap_processor = ShapProcessor(
            var_cfg=self.var_cfg,
            processed_output_path=self.processed_output_path,
            name_mapping=self.name_mapping,
        )
        self.plotter = ResultPlotter(
            var_cfg=self.var_cfg,
            cfg_postprocessing=self.cfg_postprocessing,
            plot_base_dir=self.processed_output_path
        )
        self.sanity_checker = SanityChecker(
            logger=self.logger,
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            cfg_postprocessing=self.cfg_postprocessing,
            plotter=self.plotter,
        )

    def apply_methods(self) -> None:
        """
        Executes the methods specified in the cfg dynamically.

        Raises:
            ValueError: If a specified method does not exist in the class.
        """
        for method_name in self.cfg_postprocessing["method_to_apply"]:
            if not hasattr(self, method_name):
                raise ValueError(f"Method '{method_name}' is not implemented.")

            getattr(self, method_name)()

    def condense_cv_results(self) -> None:
        """

        Returns:

        """
        cv_results_filename = self.cfg_postprocessing["general"]["filenames"]["cv_results_summarized"]

        cfg_condense_results = self.cfg_postprocessing["condense_cv_results"]
        metrics = cfg_condense_results["metrics"]
        store_all_results = cfg_condense_results["all_results"]["store"]
        all_results_filename = cfg_condense_results["all_results"]["filename"]

        cv_results_dct = self.data_loader.extract_cv_results(
            base_dir=self.processed_output_path,
            metrics=metrics,
            cv_results_filename=cv_results_filename
        )
        if store_all_results:
            all_result_path = os.path.join(self.processed_output_path, all_results_filename)
            self.data_saver.save_json(cv_results_dct, all_result_path)

        for crit, crit_vals in cv_results_dct.items():
            for samples_to_include, samples_to_include_vals in crit_vals.items():
                for nnse_analysis in [True, False]:

                    # Skip certain combinations that do not exist
                    if nnse_analysis and samples_to_include == "control":
                        continue

                    data_for_table = merge_M_SD_in_dct(samples_to_include_vals)

                    self.result_table_creator.create_cv_results_table(
                        crit=crit,
                        samples_to_include=samples_to_include,
                        data=data_for_table,
                        output_dir=self.processed_output_path,
                        nnse_analysis=nnse_analysis
                    )
                    
    def sanity_check_pred_vs_true(self) -> None:
        """

        Returns:

        """
        self.sanity_checker.sanity_check_pred_vs_true()
        
    def create_descriptives(self) -> None:
        """

        Returns:

        """
        desc_cfg = self.cfg_postprocessing["create_descriptives"]
        var_table_cfg = desc_cfg["m_sd_table"]

        self.descriptives_creator.create_m_sd_var_table(
            vars_to_include=var_table_cfg["vars_to_include"],
            binary_stats_to_calc=var_table_cfg["bin_agg_lst"],
            continuous_stats_to_calc=var_table_cfg["cont_agg_dct"],
            table_decimals=var_table_cfg["decimals"],
            store_table=var_table_cfg["store"],
            filename=var_table_cfg["filename"],
            store_index=var_table_cfg["store_index"]
        )

        rel_dct = {}

        for dataset in self.datasets:
            traits_base_filename = self.cfg_postprocessing["create_descriptives"]["traits_base_filename"]
            path_to_trait_df = os.path.join(self.data_base_path, f"{traits_base_filename}_{dataset}")
            trait_df = self.data_loader.read_pkl(path_to_trait_df) if os.path.exists(path_to_trait_df) else None

            states_base_filename = self.cfg_postprocessing["create_descriptives"]["states_base_filename"]
            path_to_state_df = os.path.join(self.data_base_path, f"{states_base_filename}_{dataset}")
            state_df = self.data_loader.read_pkl(path_to_state_df) if os.path.exists(path_to_state_df) else None
            esm_id_col = self.var_cfg["preprocessing"]["esm_id_col"][dataset]
            esm_tp_col = self.var_cfg["preprocessing"]["esm_timestamp_col"][dataset]

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
                trait_corr=wb_items_dct["trait_corr"]
            )

        if desc_cfg["rel"]["store"]:
            file_path = os.path.join(
                self.descriptives_creator.desc_results_base_path,
                desc_cfg["rel"]["filename"]
            )
            self.data_saver.save_json(rel_dct, file_path)

    def conduct_significance_tests(self) -> None:
        """

        Returns:

        """
        self.significance_testing.significance_testing()

    def create_cv_results_plot(self) -> None:
        """

        Returns:

        """
        # TODO adjust
        all_results_file_path = os.path.join(self.processed_output_path, "all_cv_results")
        with open(all_results_file_path, 'r') as infile:
            all_cv_results_dct = json.load(infile)

        if all_cv_results_dct:
            self.plotter.plot_cv_results_plots_wrapper(
                cv_results_dct=all_cv_results_dct,
                rel=None,
            )

    def create_shap_plots(self) -> None:
        """

        Returns:

        """
        # TODO Adjust
        self.plotter.plot_shap_beeswarm_plots(
            prepare_shap_data_func=self.shap_processor.prepare_shap_data,
            prepare_shap_ia_data_func=self.shap_processor.prepare_shap_ia_data,
        )

    def calculate_exp_lin_models(self) -> None:
        """

        Returns:

        """
        # TODO: Adjust -> Set full_data and all_cv_results as postprocessor attributes!
        df = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        for feature_combination in self.var_cfg["postprocessing"]["linear_regressor"]["feature_combinations"]:
            for samples_to_include in self.var_cfg["postprocessing"]["linear_regressor"]["samples_to_include"]:
                for crit in self.var_cfg["postprocessing"]["linear_regressor"]["crits"]:
                    for model_for_features in self.var_cfg["postprocessing"]["linear_regressor"]["models"]:
                        linearregressor = LinearRegressor(
                            var_cfg=self.var_cfg,
                            processed_output_path=self.processed_output_path,
                            df=df,
                            feature_combination=feature_combination,
                            crit=crit,
                            samples_to_include=samples_to_include,
                            model_for_features=model_for_features,
                            meta_vars=self.meta_vars
                        )

                        linearregressor.get_regression_data()
                        lin_model = linearregressor.compute_regression_models()

                        self.result_table_creator.create_coefficients_table(
                            model=lin_model,
                            feature_combination=feature_combination,
                            output_dir=os.path.join(self.processed_output_path, "tables")
                        )
    def create_lin_model_coefs_supp(self) -> None:
        """

        Returns:

        """
        # TODO Adjust
        self.create_mirrored_dir_with_files(
            base_dir="../results/run_2012",
            file_name="lin_model_coefs_summary.json",
            output_base_dir="../results/run_2012_lin_model_coefs",
        )

    def create_shap_values_supp(self) -> None:
        """

        Returns:

        """
        # TODO Adjust
        self.create_mirrored_dir_with_files(
            base_dir="../results/run_2012",
            file_name="shap_values_summary.pkl",
            output_base_dir="../results/run_2012_shap_values"
        )

    def create_shap_ia_values_supp(self) -> None:
        """

        Returns:

        """
        # TODO Adjust
        self.create_mirrored_dir_with_files(
            base_dir="../results/ia_values_0912",
            file_name="shap_ia_values_summary.pkl",
            output_base_dir="../results/run_2012_shap_ia_values"
        )

    def create_mirrored_dir_with_files(
            self,
            base_dir: str,
            file_name: str,
            output_base_dir: str,
    ) -> None:
        """
        """
        for root, _, files in os.walk(base_dir):
            if file_name in files:
                relative_path = os.path.relpath(root, base_dir)
                target_dir = os.path.join(output_base_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(target_dir, file_name)

                if file_name.startswith("lin_model_coefs"):
                    self.process_lin_model_coefs_for_supp(input_file_path, output_file_path)
                elif file_name.startswith("shap_values"):
                    self.process_shap_values_for_supp(input_file_path, output_file_path)
                elif file_name.startswith("shap_ia_values"):
                    self.process_shap_ia_values_for_supp(input_file_path, output_file_path)
                else:
                    raise ValueError(f"Input file {file_name} not supported yet")

    def process_lin_model_coefs_for_supp(self, input_file_path: str, output_file_path: str):
        """
        Updates the feature names in the linear model coefficients with the names used in the paper.
        """
        lin_model_coefs = self.data_loader.read_json(input_file_path)

        for stat, vals in lin_model_coefs.items():
            new_feature_names = apply_name_mapping(
                features=list(vals.keys()),
                name_mapping=self.name_mapping,
                prefix=True
            )
            # Replace old names with new_feature_names while maintaining the values
            updated_vals = {new_name: vals[old_name] for old_name, new_name in zip(vals.keys(), new_feature_names)}

            lin_model_coefs[stat] = updated_vals

        # Save the transformed content back to a file
        with open(output_file_path, 'w') as outfile:
            json.dump(lin_model_coefs, outfile, indent=4)

    def process_shap_values_for_supp(self, input_file_path: str, output_file_path: str):
        """

        Args:
            input_file_path:
            output_file_path:

        Returns:

        """
        shap_values = self.data_loader.read_pkl(input_file_path)
        feature_names_copy = shap_values["feature_names"].copy()
        feature_names_copy = [feature for feature in feature_names_copy if feature not in self.meta_vars]
        formatted_feature_names = apply_name_mapping(
                features=feature_names_copy,
                name_mapping=self.name_mapping,
                prefix=True,
            )
        shap_values["feature_names"] = formatted_feature_names

        with open(output_file_path, "wb") as f:
            pickle.dump(shap_values, f)

    def process_shap_ia_values_for_supp(self, input_file_path: str, output_file_path: str):
        """

        Args:
            input_file_path:
            output_file_path:

        Returns:

        """
        shap_ia_values = self.data_loader.read_pkl(input_file_path)
        srmc_name_mapping = {f"srmc_{feature}": feature_formatted for feature, feature_formatted
                             in self.name_mapping["srmc"].items()}
        renamed_ia_values = self.rename_srmc_keys(shap_ia_values, srmc_name_mapping)

        with open(output_file_path, "wb") as f:
            pickle.dump(renamed_ia_values, f)

    def rename_srmc_keys(self, data, srmc_name_mapping):
        """
        Recursively traverse a nested dictionary and rename:
          - single string keys that start with 'srmc'
          - tuple-of-string keys if any string within starts with 'srmc'
        using the given srmc_name_mapping.
        """
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():

                # --- Determine the "new_key" based on whether key is string or tuple ---
                if isinstance(key, str) and key.startswith("srmc"):
                    # Single string key
                    new_key = srmc_name_mapping.get(key, key)

                elif isinstance(key, tuple):
                    # Tuple of keys
                    replaced_tuple = []
                    for part in key:
                        if isinstance(part, str) and part.startswith("srmc"):
                            replaced_tuple.append(srmc_name_mapping.get(part, part))
                        else:
                            replaced_tuple.append(part)
                    new_key = tuple(replaced_tuple)

                else:
                    # For any other kind of key, or non-srmc string
                    new_key = key

                # Recursively process the value
                new_dict[new_key] = self.rename_srmc_keys(value, srmc_name_mapping)

            return new_dict

        # If `data` is not a dict, just return it as-is (e.g., int, str, list, etc.)
        return data



