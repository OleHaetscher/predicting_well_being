import os

import numpy as np
import pandas as pd
import statsmodels

from src.utils.DataSaver import DataSaver
from src.utils.utilfuncs import custom_round, NestedDict, apply_name_mapping, format_p_values


class ResultTableCreator:
    """
    This class
        - creates tables for the cv results with custom affordances
    """
    def __init__(self, fix_cfg, var_cfg, cfg_postprocessing, name_mapping):
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.cfg_postprocessing = cfg_postprocessing

        self.name_mapping = name_mapping
        self.data_saver = DataSaver()

        # Mappings for correct display in the result tables
        self.feature_combo_name_mapping_main = self.cfg_postprocessing["general"]["feature_combinations"]["name_mapping"]["main"]
        self.feature_combo_name_mapping_supp = self.cfg_postprocessing["general"]["feature_combinations"]["name_mapping"]["supp"]

        self.samples_to_include_name_mapping = self.cfg_postprocessing["general"]["samples_to_include"]["name_mapping"]
        self.crit_name_mapping = self.cfg_postprocessing["general"]["crits"]["name_mapping"]
        self.model_name_mapping = self.cfg_postprocessing["general"]["models"]["name_mapping"]
        self.metric_name_mapping = self.cfg_postprocessing["general"]["metrics"]["name_mapping"]

    def create_cv_results_table(
            self,
            data: NestedDict,
            crit: str,
            samples_to_include: str,
            output_dir: str,
            nnse_analysis: bool = False
    ) -> None:
        """
        Generates and saves a cross-validation (CV) results table as an Excel file.

        The table organizes results by:
            - Rows: Feature combinations.
            - Columns: A MultiIndex with:
                - Level 1: Models (e.g., ENR, RFR).
                - Level 2: Metrics (e.g., Pearson's r, R², Spearman's rho, MSE).
                - Level 3: Statistics (e.g., Mean (M), Standard Deviation (SD)).

        The function filters and maps feature combinations, aggregates statistics, and ensures
        a consistent structure for display.

        Args:
            data: Nested dictionary of CV results with the structure:
                  {feature_combination: {model: {metric: {'M': float, 'SD': float}}}}.
            crit: Criterion used for the table (e.g., "wb_state").
            samples_to_include: Subset of samples (e.g., "all").
            output_dir: Directory for saving the Excel file.
            nnse_analysis: Whether to include only the supplementary (nnse) analysis. Defaults to False.

        """
        cfg = self.cfg_postprocessing["condense_cv_results"]["result_table"]

        if nnse_analysis:
            feature_combo_mapping = self.feature_combo_name_mapping_supp
            result_str = cfg["result_strs"]["nnse"]
        else:
            feature_combo_mapping = self.feature_combo_name_mapping_main
            result_str = cfg["result_strs"]["main"]

        #feature_combo_mapping = {
        #    k: v
        #    for k, v in self.feature_combo_name_mapping.items()
        #    if ("nnse" in k) == nnse_analysis
        #}

        #result_str = cfg["result_strs"]["nnse" if nnse_analysis else "main"]

        # Custom order for metrics
        metric_order = ["r", "R2", "rho", "MSE"]

        # Generate rows for the DataFrame
        rows = [
            {
                "feature_combination": feature_combo_mapping[feature_combination],
                "model": self.model_name_mapping.get(model, model),
                "metric": self.metric_name_mapping.get(metric, metric),
                "M (SD)": stats.get("M (SD)", "N/A"),
            }
            for feature_combination, models in data.items()
            if feature_combination in feature_combo_mapping
            for model, metrics in models.items()
            for metric, stats in metrics.items()
        ]

        # Create a DataFrame
        df = pd.DataFrame(rows)

        # Pivot the DataFrame to create the desired structure
        df_pivot = df.pivot(index="feature_combination", columns=["model", "metric"], values="M (SD)")

        # Reorder columns by metric custom order
        df_pivot.columns = pd.MultiIndex.from_tuples(
            sorted(
                df_pivot.columns,
                key=lambda col: (
                    col[0],  # Model
                    metric_order.index(col[1]) if col[1] in metric_order else len(metric_order)  # Custom metric order
                )
            )
        )

        # Add an empty column with NaN values
        empty_col = pd.Series([np.nan] * len(df_pivot), name=(" ", " "))  # Single-level column
        df_pivot = pd.concat([df_pivot.iloc[:, :4], empty_col, df_pivot.iloc[:, 4:]], axis=1)

        # Reindex rows to ensure the feature combinations are in a custom order
        custom_order = [feature_combo_mapping[k] for k in feature_combo_mapping]
        df_pivot = df_pivot.reindex(custom_order, fill_value="N/A")

        if cfg["store"]:
            output_path = os.path.join(
                output_dir,
                f"{cfg['file_base_name']}_{crit}_{samples_to_include}_{result_str}.xlsx",
            )
            self.data_saver.save_excel(df_pivot, output_path)

        print(f"Processed {crit}_{samples_to_include}_{result_str}")


    def create_coefficients_table(
            self,
            feature_combination: str,
            model,  # Fitted statsmodels object
            output_dir: str,
    ) -> None:
        """
        Creates an R-style regression table from a fitted statsmodels object and saves it to an Excel file.

        The table includes:
        - Predictors (independent variables)
        - Estimates (coefficients)
        - Confidence Intervals (CI)
        - P-values
        - Model summary statistics (e.g., R² and Adjusted R², number of observations)

        Args:
            feature_combination (str): The name of the feature combination for the table title.
            model (Any): A fitted statsmodels regression object.
            output_dir (str): Directory to save the resulting Excel file.

        Returns:
            None: Saves the table to an Excel file.
        """
        # Extract coefficients, confidence intervals, and p-values
        coefficients = model.params
        conf_int = model.conf_int()
        p_values = model.pvalues

        # Create a DataFrame for the regression table
        regression_table = pd.DataFrame({
            'Predictors': coefficients.index,
            'Estimates': coefficients.values,
            'CI': conf_int.apply(lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]", axis=1),
            'p': p_values.values
        })

        # Round estimates and p-values for formatting
        regression_table['Estimates'] = regression_table['Estimates'].round(3)
        #regression_table['p'] = regression_table['p'].apply(
        #    lambda x: f"<0.001" if x < 0.001 else f"{x:.3f}"
        #)
        regression_table['p'] = regression_table['p'].apply(format_p_values)
        regression_table["Predictors"] = apply_name_mapping(
            features=list(regression_table["Predictors"]),
            name_mapping=self.name_mapping,
            prefix=True,
        )

        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        observations = model.nobs

        # Add a footer row for model statistics
        footer_rows = pd.DataFrame({
            'Predictors': ['Observations', 'R² / R² adjusted'],
            'Estimates': [f"{int(observations)}", f"{r_squared:.3f} / {adj_r_squared:.3f}"],
            'CI': [None, None],
            'p': [None, None]
        })

        # Combine the regression table with footer rows
        final_table = pd.concat([regression_table, footer_rows], ignore_index=True)

        output_path = os.path.join(output_dir, f'reg_table_{feature_combination}.xlsx')
        self.data_saver.save_excel(df=final_table, output_path=output_path, index=False)
