import os

import numpy as np
import pandas as pd
import statsmodels

from src.utils.utilfuncs import custom_round, NestedDict


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

        # Mappings for correct display in the result tables
        self.feature_combo_name_mapping = self.cfg_postprocessing["general"]["feature_combinations"]["name_mapping"]
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
        Creates a DataFrame containing cross-validation (CV) results for a given criterion and sample subset.

        This method processes the provided data and generates a table where:
            - Rows represent feature combinations.
            - Columns are a MultiIndex with:
                - Level 1: Models (i.e., ENR, RFR).
                - Level 2: Metrics (i.e., Pearson's r, R², Spearman's rho, MSE).
                - Level 3: Statistics (i.e., M, SD).

        The resulting DataFrame is saved as an Excel file in the specified output directory.

        Args:
            data: A nested dictionary containing CV results with the structure:
                {feature_combination: {model: {metric: {'M': float, 'SD': float}}}}.
            crit: Criterion used for the heatmap title (e.g., "wb_state").
            samples_to_include: Sample subset used (e.g., "all").
            output_dir: Directory to save the resulting Excel file.
            nnse_analysis: If True, include only the supplementary analysis without the Neuroticism facets and self-esteem.
                If False, include only the main analysis.

        """
        # Map feature combinations and filter if a subset is provided
        if nnse_analysis:
            feature_combo_mapping = (
                {k: v for k, v in self.feature_combo_name_mapping.items() if "nnse" in k}
            )
            result_str = "nnse"

        else:  # main analysis
            feature_combo_mapping = (
                {k: v for k, v in self.feature_combo_name_mapping.items() if "nnse" not in k}
            )
            result_str = "main"

        # Create a list of rows for DataFrame construction
        rows = [
            {
                'feature_combination': feature_combo_mapping[feature_combination],
                'model': self.model_name_mapping.get(model, model),
                'metric': self.metric_name_mapping.get(metric, metric),
                'M': stats.get('M', np.nan),
                'SD': stats.get('SD', np.nan)
            }
            for feature_combination, models in data.items() if feature_combination in feature_combo_mapping
            for model, metrics in models.items()
            for metric, stats in metrics.items()
        ]
        try:

            # Create and pivot the DataFrame
            df = pd.DataFrame(rows)
            df_pivot = df.pivot(index='feature_combination', columns=['model', 'metric'], values=['M', 'SD'])

            # Reorder levels and sort by model and custom metric order
            df_pivot = df_pivot.reorder_levels(['model', 'metric', None], axis=1)
            col_index_df = pd.DataFrame(df_pivot.columns.tolist(), columns=['model', 'metric', 'stat'])
            col_index_df['metric'] = pd.Categorical(
                col_index_df['metric'], categories=self.metric_name_mapping.values(), ordered=True
            )

            sorted_columns = pd.MultiIndex.from_frame(col_index_df.sort_values(by=['model', 'metric', 'stat']))
            df_pivot.columns = sorted_columns

            # Reindex rows based on feature_combo_mapping
            custom_order = [feature_combo_mapping[k] for k in feature_combo_mapping]
            df_pivot = df_pivot.reindex(custom_order, fill_value=np.nan)
            df_pivot = df_pivot.round(3)

            # Save to Excel
            output_path = os.path.join(output_dir, f'cv_results_{crit}_{samples_to_include}_{result_str}.xlsx')
            df_pivot.to_excel(output_path, merge_cells=True)

            print(f"Saved results to {output_path}")

        except:
            print(f"Not worked for {crit} - {samples_to_include} - {result_str}")

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
        regression_table['p'] = regression_table['p'].apply(
            lambda x: f"<0.001" if x < 0.001 else f"{x:.3f}"
        )

        # Extract summary statistics
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

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'reg_table_{feature_combination}.xlsx')
        final_table.to_excel(output_path, index=False, sheet_name="Regression Table")
        print(f"Saved regression table to {output_path}")
