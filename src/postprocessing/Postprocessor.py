import json
import os
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
from src.utils.DataSelector import DataSelector
from src.utils.Logger import Logger
from src.utils.SanityChecker import SanityChecker
from src.utils.utilfuncs import custom_round, apply_name_mapping, defaultdict_to_dict


class Postprocessor:
    """
    This class executes the different postprocessing steps. This includes
        - conducting tests of significance to compare prediction results
        - calculate and display descriptive statistics
        - creates plots (results, SHAP, SHAP interaction values)
    """

    def __init__(self, fix_cfg, var_cfg, cfg_postprocessing, name_mapping):
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.cfg_postprocessing = cfg_postprocessing
        self.name_mapping = name_mapping

        self.result_filenames = self.var_cfg["analysis"]["output_filenames"]
        self.processed_output_path = self.var_cfg["postprocessing"]["processed_results_path"]
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

        self.logger = Logger(
            log_dir=self.var_cfg["general"]["log_dir"],
            log_file=self.var_cfg["general"]["log_name"]
        )
        self.descriptives_creator = DescriptiveStatistics(
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            name_mapping=name_mapping
        )
        self.result_table_creator = ResultTableCreator(
            fix_cfg=self.fix_cfg,
            var_cfg=self.var_cfg,
            cfg_postprocessing=self.cfg_postprocessing,
            name_mapping=name_mapping
        )

        self.significance_testing = SignificanceTesting(
            var_cfg=self.var_cfg,
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
            plotter=self.plotter,
        )

    def postprocess(self):
        """
        This is a wrapper method that does all the postprocessing steps specified in the config.
        It may invoke the methods in the other classes. What it does:

            sanity_check_pred_vs_true:
                Make post-hoc analyses based on the predicted and the true criterion values
            condense_cv_results:
                Creates tables containing all results for different metrics
            condense_lin_model_coefs:
                Creates tables containing the ENR coeffiecients for all analyses
            create_descriptives:
                Calculates descriptive statistics
                    - M/SD of features, wb_items, and criteria
                    - ICCs and bp/wp correlation of wb_items
                    - reliability of criteria per sample
            conduct_significance_tests:
                Conducts significance tests (corrected paired t-tests) and apply FDR correction
            create_cv_results_plot:
                Creates a plot that visualizes the prediction results for both models and all feature combinations
            create_shap_plots:
                Creates plots visualizing SHAP and SHAP interaction values. This includes
                    SHAP beeswarm plots representing the most imortant features for all feature combinations
                    SHAP importance plots TBA
        """
        if "calculate_lin_models" in self.methods_to_apply:
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

        if "sanity_check_pred_vs_true" in self.methods_to_apply:
            self.sanity_checker.sanity_check_pred_vs_true()

        if "condense_cv_results" in self.methods_to_apply:
            # TODO We change this, as we get the metric together in one table and not separately
            cv_results_dct = self.data_loader.extract_cv_results(
                base_dir=self.processed_output_path,
                metrics=self.cfg_postprocessing["condense_cv_results"]["metrics"],
                cv_results_filename=self.cfg_postprocessing["filenames"]["cv_results_summarized"]
            )
            with open(os.path.join(self.processed_output_path, "all_cv_results"), 'w') as outfile:
                json.dump(cv_results_dct, outfile, indent=4)

            # Create separate tables for each crit and each samples to include
            for crit, crit_vals in cv_results_dct.items():
                for samples_to_include, samples_to_include_vals in crit_vals.items():
                    for bool_val in [True, False]:
                        self.result_table_creator.create_cv_results_table(
                            crit=crit,
                            samples_to_include=samples_to_include,
                            data=samples_to_include_vals,
                            output_dir=self.processed_output_path,
                            nnse_analysis=bool_val  # nnse or not
                        )

        if "condense_lin_model_coefs" in self.methods_to_apply:
            # store coefficients in a separate directory -> as file to supplement # TODO move filenames to cfg_postprocessing
            self.create_lin_model_coefs_dir(
                base_dir="../results/run_2012",
                file_name="lin_model_coefs_summary.json",
                output_base_dir="../results/run_2012_lin_model_coefs"
            )

        if "create_descriptives" in self.methods_to_apply:
            self.descriptives_creator.create_m_sd_feature_table()
            self.descriptives_creator.create_crit_table()
            print()

            # Also a function that creates
            for dataset in self.datasets:  # TODO Integrate in criteria table, if for crit?
                state_rel_series, trait_rel_series = self.descriptives_creator.compute_rel(dataset=dataset)
                wb_items_dct = self.descriptives_creator.create_wb_items_stats_per_dataset(
                    dataset=dataset
                )
                self.descriptives_creator.create_wb_items_table(
                    dataset=dataset,
                    m_sd_df=wb_items_dct["m_sd"],
                    icc1=wb_items_dct["icc1"],
                    icc2=wb_items_dct["icc2"],
                    bp_corr=wb_items_dct["bp_corr"],
                    wp_corr=wb_items_dct["wp_corr"],
                    trait_corr=wb_items_dct["trait_corr"]
                )

        if "conduct_significance_tests" in self.methods_to_apply:
            self.significance_testing.significance_testing()  # (dct=self.cv_results_dct.copy())

        if "create_cv_results_plots" in self.methods_to_apply:
            all_results_file_path = os.path.join(self.processed_output_path, "all_cv_results")
            with open(all_results_file_path, 'r') as infile:
                all_cv_results_dct = json.load(infile)

            if all_cv_results_dct:
                self.plotter.plot_cv_results_plots_wrapper(
                    cv_results_dct=all_cv_results_dct,
                    rel=None,
                )

        if "create_shap_plots" in self.methods_to_apply:
            # self.get_n_samples_per_analysis()

            self.plotter.plot_shap_beeswarm_plots(
                prepare_shap_data_func=self.shap_processor.prepare_shap_data,
                prepare_shap_ia_data_func=self.shap_processor.prepare_shap_ia_data,
                # n_samples_dct=self.n_samples_dct,
            )

    def get_n_samples_per_analysis(self) -> None:
        """
        Extracts the number of samples per analysis for correct display in the plots.

        This method creates a Dict that contains the number of samples for all
        feature_combination / samples_to_include / crit combinations and sets it as an attribute.

        Sets:
            xxx
        """
        full_data = pd.read_pickle(os.path.join(self.var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))
        n_samples_dct = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        feature_combinations = self.cfg_postprocessing["general"]["feature_combinations"]["name_mapping"].keys()
        samples_to_include_options = self.cfg_postprocessing["general"]["samples_to_include"]["name_mapping"].keys()
        crits = self.cfg_postprocessing["general"]["crits"]["name_mapping"].keys()

        for feature_comb in feature_combinations:
            for samples_to_include in samples_to_include_options:
                for crit in crits:

                    data_selector = DataSelector(
                        var_cfg=self.var_cfg,
                        df=full_data.copy(),
                        feature_combination=feature_comb,
                        crit=crit,
                        samples_to_include=samples_to_include,
                        meta_vars=self.meta_vars
                    )
                    try:
                        data_selector.select_samples()
                    except KeyError:
                        print("No data for", feature_comb, samples_to_include, crit)
                    n_samples_dct[feature_comb][samples_to_include][crit] = len(data_selector.df)

        n_samples_dct = defaultdict_to_dict(n_samples_dct)

        self.n_samples_dct = n_samples_dct

    @staticmethod
    def create_df_table(data, metric, output_dir, feature_combo_mapping):
        """
        Create DataFrame from metrics data, save to Excel.

        Args:
            data (list): Data points extracted for DataFrame.
            metric (str): The metric used for the heatmap title.
            output_dir (str): Directory to save the Excel file.
            custom_order (list): The desired order of feature combinations for the top-level columns.
            feature_combo_mapping (dict): Mapping from feature_combination keys to descriptive labels.
        """
        if not data:
            return  # No data to process

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Map the feature_combination column to meaningful labels
        df['feature_combination'] = df['feature_combination'].map(feature_combo_mapping)

        custom_order = list(feature_combo_mapping.values())
        custom_order = [fc for fc in custom_order
                        if "Interaction Values" not in fc]

        df["model"] = df["model"].map(
            {"elasticnet": "ENR", "randomforestregressor": "RFR"}
        )

        df["crit"] = df["crit"].map(
            {
            "wb_state": "Experienced well-being",
            "wb_trait": "Remembered well-being",
            "pa_state": "Experienced positive affect",
            "na_state": "Experienced negative affect",
            "pa_trait": "Remembered positive affect",
            "na_trait": "Remembered positive affect",
            }
        )
        df["samples_to_include"] = df["samples_to_include"].map(
            {
            "w": "Experienced well-being",
            "wb_trait": "Remembered well-being",
            "pa_state": "Experienced positive affect",
            "na_state": "Experienced negative affect",
            "pa_trait": "Remembered positive affect",
            "na_trait": "Remembered positive affect",
            }
        )

        # Set multi-index
        df.set_index(['crit', 'model', 'samples_to_include'], inplace=True)

        # Pivot for mean (M) and std (SD)
        df_mean = df.pivot_table(
            values=f"m_{metric}",
            index=['crit', 'model', 'samples_to_include'],
            columns='feature_combination',
            aggfunc=np.mean
        )
        df_sd = df.pivot_table(
            values=f"sd_{metric}",
            index=['crit', 'model', 'samples_to_include'],
            columns='feature_combination',
            aggfunc=np.mean
        )

        # Concatenate mean and SD, creating a multi-index with levels ['M','SD']
        combined_df = pd.concat([df_mean, df_sd], keys=['M', 'SD'], axis=1)

        # By default, the new columns have a MultiIndex of the form: (M, fc1), (M, fc2), ... (SD, fc1), (SD, fc2)...
        # We want the top level to be feature_combinations and the second level to be M / SD.
        # So we reorder levels: [1, 0] means "feature_combination" on top, then M/SD as second.
        combined_df = combined_df.reorder_levels([1, 0], axis=1)

        # Now explicitly build the column order. For each feature combination in custom_order, we want (fc, 'M') then (fc, 'SD').
        desired_cols = []
        for fc in custom_order:
            desired_cols.append((fc, 'M'))
            desired_cols.append((fc, 'SD'))

        # Reindex the DataFrame to enforce that column order.
        # Columns not in `desired_cols` will be dropped; columns in `desired_cols` but not present will become NaN.
        combined_df = combined_df.reindex(columns=desired_cols)

        # Round to three decimals
        # combined_df = combined_df.round(3)
        combined_df = combined_df.applymap(lambda x: custom_round(x, 3) if not np.isnan(x) else x)


        combined_df = combined_df.T

        output_path = os.path.join(output_dir, f'cv_results_{metric}.xlsx')
        combined_df.to_excel(output_path, merge_cells=True)

    def create_lin_model_coefs_dir(
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

                # Load JSON content
                with open(input_file_path, 'r') as infile:
                    lin_model_coefs = json.load(infile)

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




