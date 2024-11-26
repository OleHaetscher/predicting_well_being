import os
import random
import string
from itertools import product
from typing import Callable

from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import shap

import matplotlib.colors as mcolors
import colorsys


class ResultPlotter:
    """
    This class plots the results. It should always get the data that is readily processed, so that this class
    really only handles the plotting of the data.
    This includes
        - Setting up a baseplot
        - Setting up the subplots
        - Plotting the data
        - Adjusting the format
    """
    def __init__(self, var_cfg, plot_base_dir):
        self.var_cfg = var_cfg
        self.plot_base_dir = os.path.join(plot_base_dir, "plots")  # this is the folder containing the processed final results
        self.store_plots = self.var_cfg["postprocessing"]["plots"]["store_plots"]

    def plot_figure_2(self, data_to_plot: dict, rel: float):
        """
        Function to create a figure with three columns, each with multiple rows representing a different
        set of feature combinations.
        """
        for samples_to_include in self.var_cfg["postprocessing"]["plots"]["cv_result_plot"]["samples_to_include"]:
            for crit in self.var_cfg["postprocessing"]["plots"]["cv_result_plot"]["crit"]:

                filtered_metrics = {
                    key.replace(crit + "_", "").replace("_" + samples_to_include, ""): value
                    for key, value in data_to_plot.items()
                    if key.startswith(crit) and key.endswith(samples_to_include)
                }
                ##### Manually replace some filtered metrics
                # pl_srmc_sens -> Use all Analysen: RFR: Mean = 0.607, SD = 0.0207, ENR: 0.597, 0.0213
                filtered_metrics["pl_srmc_sens_elasticnet"] = {"m_spearman": 0.597, 'sd_spearman': 0.0213}
                filtered_metrics["pl_srmc_sens_randomforestregressor"] = {"m_spearman": 0.607, 'sd_spearman': 0.0207}
                filtered_metrics["pl_sens_elasticnet"] = {"m_spearman": 0.570, 'sd_spearman': 0.0213}
                filtered_metrics["pl_sens_randomforestregressor"] = {"m_spearman": 0.573, 'sd_spearman': 0.0207}
                pl_margin_dct = self.compute_pl_margin(filtered_metrics)

                # color_dct = {"pl": "#5E99CB", "srmc": "#76ACBE", "sens": "#A1D2A7", "mac": "#E6F3E8"}
                # color_dct = {"pl": "#5E99CB", "srmc": "#76ACBE", "sens": "#76ACBE", "mac": "#E6F3E8"}
                color_dct = {"pl": "#5E9ACC", "srmc": "#A3C7A1", "sens": "#A3C7A1", "mac": "#F0E68C"}
                fig, axes = self.create_grid(num_cols=3, num_rows=4, figsize=(20, 10), empty_cells=[(3, 2), (2, 2)])
                models = ["elasticnet", "randomforestregressor"]
                x_min, x_max = -0.1, 1

                # Define groups and titles for each column
                fig_groups = [["pl", "srmc", "sens", "mac"],  # Each row in this list is a single category
                              ["pl_srmc", "pl_sens", "pl_srmc_sens", "pl_mac"],
                              ["pl_srmc_mac", "all_in"]]
                feature_combo_mapping = {
                    "pl": "Person-level",
                    "srmc": "ESM",
                    "sens": "Sensing",
                    "mac": "Macro-level",
                    "pl_srmc": "Person-level + ESM",
                    "pl_sens": "Person-level + Sensing",
                    "pl_srmc_sens": "Person-level + ESM + Sensing",
                    "pl_mac": "Person-level + Macro-level",
                    "srmc_sens": "ESM + Sensing",
                    "srmc_mac": "ESM + Macro-level",
                    "sens_mac": "Sensing + Macro-level",
                    "pl_srmc_mac": "Person-level + ESM + Macro-level",
                    "pl_sens_mac": "Person-level + Sensing + Macro-level",
                    "srmc_sens_mac": "ESM + Sensing + Macro-level",
                    "pl_srmc_sens_mac": "Person-level + ESM + Sensing + Macro-level",
                    "all_in": "All features"
                }

                significance_results_models = [[0.49, 0.03, 0.58, 0.39],
                                                [0.49, 0.03, 0.58, 0.39],
                                                [0.49, 0.30]]
                # fix later
                titles = ["Within Conceptual Levels", "Across Two Conceptual Levels", "Across Three Conceptual Levels"]

                # Prepare filtered metrics for each group
                filtered_metrics_col = [
                    {key: value for key, value in filtered_metrics.items() if
                     key in [f"{prefix}_{model}" for prefix, model in product(group, models)]}
                    for group in fig_groups
                ]
                pl_ref_dct = {"elasticnet": filtered_metrics_col[0]["pl_elasticnet"],
                              "randomforestregressor": filtered_metrics_col[0]["pl_randomforestregressor"]}

                # Loop through each group and plot in the respective column
                for i, (group, title, metrics) in enumerate(zip(fig_groups, titles, filtered_metrics_col)):
                    for j, category in enumerate(group):  # Each row gets one category
                        ax = axes[j, i]
                        single_category_metrics = {
                            f"{category}_{model}": metrics[f"{category}_{model}"] for model in models if
                            f"{category}_{model}" in metrics
                        }

                        if i == 0:
                            # Plot for the first column (Within Conceptual Levels)
                            self.plot_bar_plot(ax=ax, data=single_category_metrics, order=[category], models=models,
                                               color_dct=color_dct, feature_combo_mapping=feature_combo_mapping, rel=rel)
                        elif i == 1:
                            # Plot for the second column (Across Two Conceptual Levels)
                            self.plot_incremental_bar_plot(ax=ax, data=single_category_metrics, pl_margin_dct=pl_margin_dct,
                                                           order=[category],feature_combo_mapping=feature_combo_mapping,
                                                           models=models, color_dct=color_dct, rel=rel, pl_reference_dct=pl_ref_dct)
                        elif i == 2 and j < 2:
                            # Plot in the third column only for the first 2 rows
                            pass
                            # self.plot_bar_plot(ax=ax, data=single_category_metrics, order=[category], models=models,
                                               # color_dct=color_dct, rel=rel)

                        ax.set_xlim(x_min, x_max)
                        ax.set_title(title if j == 0 else "", fontsize=15, pad=15, fontweight="bold")
                        ax.tick_params(axis='x', labelsize=12)
                        ax.tick_params(axis='y', labelsize=12)

                # Add a general legend in the bottom right
                # fig.legend(models, loc='lower right', bbox_to_anchor=(0.85, 0.05), ncol=1, title="Models", frameon=False)
                # Define the legend elements
                randomforest_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='RandomForestRegressor (Upper Bar)')
                elasticnet_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='ElasticNet (Lower Bar)')
                reliability_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='Reliability = .90')

                # Add the custom legend to the figure
                fig.legend(handles=[randomforest_patch, elasticnet_patch, reliability_line],
                           loc='lower right', bbox_to_anchor=(0.98, 0.15), ncol=1, frameon=False, title="", fontsize=15)

                # Adjust layout for readability
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                if self.store_plots:
                    self.store_plot(plot_name="cv_results", crit=crit, samples_to_include=samples_to_include, model=None)
                else:
                    plt.show()
    def plot_bar_plot(self, ax, data, order, models, color_dct, feature_combo_mapping, rel=0.90):
        """
        Plots a bar plot for the given data with error bars, with each feature group displayed separately.
        Uses different saturation levels to differentiate models and adds a vertical line for `rel`.

        Args:
            ax: matplotlib axis object to plot on.
            data: dict, where keys are feature-model combinations (e.g., 'pl_elasticnet') and values are dictionaries.
            order: list of feature combinations (e.g., ['pl', 'srmc', 'mac', 'sens']) to control y-axis order.
            models: list of models (e.g., ['elasticnet', 'randomforestregressor']) to color the bars.
            color_dct: Optional dict of colors for each feature category.
            rel: float, vertical line value to indicate a threshold.
        """
        metric = "m_spearman"
        error_metric = "sd_spearman"
        bar_width = 0.15

        # Generate bar positions for each feature group
        y_positions = np.arange(len(order))

        # Loop through each model and feature group
        for i, model in enumerate(models):
            for j, feature in enumerate(order):
                # Set color saturation to differentiate models
                base_color = color_dct[feature]
                alpha = 1 if model == "elasticnet" else 0.7

                model_key = f"{feature}_{model}"
                value = data.get(model_key, {}).get(metric, 0)
                error = data.get(model_key, {}).get(error_metric, 0)

                # Plot the bar with adjusted color
                ax.barh(y_positions[j] + (i * bar_width), value, xerr=error, height=bar_width,
                        color=base_color, align='center', capsize=5, edgecolor=None, alpha=alpha)

                # Define the coordinates for the bracket
                left, right = ax.get_xlim()
                bottom, top = ax.get_ylim()
                # may place some significance bracket here


        # Set y-ticks to the feature group labels in reversed order
        ax.set_yticks(y_positions + bar_width / 2)

        new_lst = [feature_combo_mapping[feature] for feature in order]
        ax.set_yticklabels(new_lst, fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        if "spearman" in metric:
            ax.set_xlabel(r'$\rho$', fontsize=15)
        # ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Draw the rel line
        ax.axvline(x=rel, color='black', linestyle='--', linewidth=1.2, label=f'Rel ({rel:.2f})')

    def plot_incremental_bar_plot(self,
                                  ax,
                                  data,
                                  pl_margin_dct,
                                  pl_reference_dct,
                                  order,
                                  models,
                                  color_dct,
                                  feature_combo_mapping,
                                  rel):
        """
        Plots an incremental bar plot, splitting each bar into 'pl' base and the incremental effect.

        Args:
            ax: matplotlib axis object to plot on.
            data: dict of feature-model combinations with metrics.
            pl_margin_dct: dict from compute_pl_margin, containing incremental metrics.
            order: list of feature combinations.
            models: list of models.
            color_dct: Color dictionary for each feature.
        """
        metric = "m_spearman"
        error_metric = "sd_spearman"
        bar_width = 0.3
        y_positions = np.arange(len(order))

        for i, model in enumerate(models):
            for j, feature in enumerate(order):
                # Combination, plot incremental bar
                base_value = pl_reference_dct[model][metric]
                increment = pl_margin_dct.get(f"{feature}_{model}", {}).get("incremental_m_spearman", 0)
                error = data[f"{feature}_{model}"][error_metric]

                base_feature = feature.split("_")[1]
                other_feat_color = color_dct[base_feature]
                pl_color = color_dct["pl"]
                alpha = 1 if model == "elasticnet" else 0.7

                ax.barh(y_positions[j] + (i * bar_width), base_value, height=bar_width, color=pl_color,
                        align='center', edgecolor=None, alpha=alpha)
                if increment > 0:
                    ax.barh(y_positions[j] + (i * bar_width), increment, left=base_value, height=bar_width,
                            xerr=error, color=other_feat_color, align='center', edgecolor=None, alpha=alpha,
                            capsize=5)
                else:
                    ax.barh(y_positions[j] + (i * bar_width), 0, left=base_value, height=bar_width,
                            xerr=error, color=other_feat_color, align='center', edgecolor=None, alpha=alpha, # hatch
                            capsize=5)

            ax.set_yticks(y_positions + bar_width / 2)

            new_lst = [feature_combo_mapping[feature] for feature in order]
            formatted_labels = [label.replace('+', '\n +') for label in new_lst]
            ax.set_yticklabels(formatted_labels, fontsize=15)
            ax.tick_params(axis='x', labelsize=15)
            if "spearman" in metric:
                ax.set_xlabel(r'$\rho$', fontsize=15)
            # ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Draw the rel line
            ax.axvline(x=rel, color='black', linestyle='--', linewidth=1.2, label=f'Rel ({rel:.2f})')

    def add_significance_bracket(self, ax, y1, y2, significance_level):
        """
        Adds a vertical significance bracket with asterisks to the right of two bars, using
        axis-relative positioning to avoid changing the layout.

        Args:
            ax: Matplotlib axis object where the bracket will be drawn.
            y1: The y-coordinate of the bottom error bar (relative position within axis).
            y2: The y-coordinate of the top error bar (relative position within axis).
            significance_level: p-value for determining the number of asterisks.
        """
        # Determine the symbol based on the p-value
        if significance_level < 0.001:
            sig_symbol = "***"
        elif significance_level < 0.01:
            sig_symbol = "**"
        elif significance_level <= 0.05:
            sig_symbol = "*"
        else:
            sig_symbol = ""

        if sig_symbol:  # Only add bracket if significance is <= 0.05
            # Position x just beyond the right edge in relative coordinates
            x_bracket = 1.05

            # Plot vertical line for the bracket in relative coordinates
            ax.plot([x_bracket, x_bracket], [y1, y2], lw=1.5, c='black', transform=ax.transAxes, clip_on=True)

            # Draw short horizontal lines at the ends of the vertical line
            bracket_width = 0.02  # Width of the horizontal lines in relative units
            ax.plot([x_bracket, x_bracket + bracket_width], [y1, y1], lw=1.5, c='black', transform=ax.transAxes, clip_on=True)
            ax.plot([x_bracket, x_bracket + bracket_width], [y2, y2], lw=1.5, c='black', transform=ax.transAxes, clip_on=True)

            # Place the significance symbol next to the bracket
            ax.text(x_bracket + bracket_width * 1.5, (y1 + y2) / 2, sig_symbol, ha='left', va='center',
                    color='black', fontsize=12, transform=ax.transAxes, clip_on=True)

    def create_grid(self, num_rows: int, num_cols: int, figsize=(15, 20), empty_cells=None):
        """
        This function creates a flexible grid of subplots with customizable empty cells.

        Args:
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
            figsize (tuple): Figure size for the entire grid (default is (15, 20)).
            empty_cells (list of tuples): List of (row, col) tuples where cells should be empty.

        Returns:
            fig (plt.Figure): Matplotlib figure object for the grid.
            axes (np.ndarray): Array of axes objects for individual plots.
        """
        # Initialize figure and axes grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

        # Set specified cells as empty
        if empty_cells:
            for (row, col) in empty_cells:
                fig.delaxes(axes[row, col])  # Remove the axis for empty cells

        # Return figure and axes array
        return fig, axes


    def compute_pl_margin(self, filtered_metrics):
        """
        Computes the incremental performance difference for all combinations involving 'pl' (e.g., 'pl_srmc')
        compared to the base 'pl' metric for each model.

        Args:
            filtered_metrics (dict): Dictionary containing metrics for each model and feature combination.

        Returns:
            dict: Dictionary containing incremental performance differences for each 'pl' combination and model.
        """
        # Initialize a dictionary to store the incremental performance margins
        pl_margin_dict = {}

        # Get base metrics for 'pl' only (e.g., 'pl_elasticnet' and 'pl_randomforestregressor')
        base_metrics = {key: value for key, value in filtered_metrics.items() if key.startswith("pl_") and "_" not in key[len("pl_"):]}

        # Loop over each key in the filtered_metrics that contains 'pl_' but is not the base 'pl'
        for key, value in filtered_metrics.items():
            if key.startswith("pl_") and key not in base_metrics:
                # Extract the model (e.g., 'elasticnet' or 'randomforestregressor') from the key
                model = key.split('_')[-1]
                base_key = f"pl_{model}"

                # Check if the base metric exists
                if base_key in base_metrics:
                    # Calculate the difference in metric ('m_spearman') for the combination vs the base 'pl'
                    incremental_difference = value['m_spearman'] - base_metrics[base_key]['m_spearman']

                    # Store the incremental difference with the key indicating the combination and model
                    pl_margin_dict[key] = {
                        'incremental_m_spearman': incremental_difference,
                        'sd_spearman': value['sd_spearman']  # Keep the standard deviation of the combination
                    }

        return pl_margin_dict

    def plot_shap_importance_plot(self, data: dict, crit: str, samples_to_include: str, model: str):
        """
        This function

        Args:
            data:
            crit:
            samples_to_include:
            model:

        Returns:

        """
        data_current = data[crit][samples_to_include][model]
        fig, axes = self.create_grid(num_rows=4, num_cols=3, figsize=(20, 15), empty_cells=[(2, 2), (3, 2)])

        # Define the arrangement of predictor combinations in the grid
        first_col = ["pl", "srmc", "sens", "mac"]
        second_col = ["pl_srmc", "pl_sens", "pl_srmc_sens", "pl_mac"]
        third_col = ["pl_srmc_mac", "all_in"]

        # Map predictor combinations to their positions in the grid
        positions = {}
        # First column
        for row, predictor_combination in enumerate(first_col):
            positions[(row, 0)] = predictor_combination
        # Second column
        for row, predictor_combination in enumerate(second_col):
            positions[(row, 1)] = predictor_combination
        # Third column
        for row, predictor_combination in enumerate(third_col):
            positions[(row, 2)] = predictor_combination
        # Empty positions are at (2, 2) and (3, 2)

        # Iterate over the positions and plot the SHAP importance plots
        for (row, col), predictor_combination in positions.items():
            if predictor_combination in data_current:
                shap_values = data_current[predictor_combination]

                ax = axes[row][col]
                # Plot the SHAP importance plot in the specified subplot
                shap.plots.bar(
                    shap_values,
                    max_display=6,
                    order=shap.Explanation.abs,
                    show=False,
                    ax=ax,
                )
                # Set the title of the subplot to the predictor combination name
                ax.set_title(predictor_combination)
            else:
                print(f"Predictor combination '{predictor_combination}' not found in data.")

        # Adjust layout and display the figure
        # Save the plot
        plt.tight_layout()
        plt.show()
        #plot_name = f"{crit}_{model}_imp.png"
        #plt.savefig(plot_name, format='png', dpi=300)  # You can adjust the format and DPI as needed
        #plt.close()

    def plot_shap_beeswarm_plots(self, prepare_data_func: Callable):
        """
        Plots SHAP beeswarm plots for different predictor combinations arranged in a grid.

        Args:
            prepare_data_func

        Returns:
            None
        """
        crits = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"].get("crit", [])
        samples_to_include_list = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"].get("samples_to_include", [])
        models = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"].get("prediction_model", [])

        # Iterate over each combination of crit, samples_to_include, and model
        for crit in crits:
            for samples_to_include in samples_to_include_list:
                for model in models:
                    print("### Plot combination:", samples_to_include, crit, model)
                    data_current = prepare_data_func(crit_to_plot=crit, samples_to_include=samples_to_include, model_to_plot=model,
                                                     custom_affordances={"sens": "selected", "mac": "selected"})

                    # Create a grid of subplots with specified empty cells
                    fig, axes = self.create_grid(num_rows=3, num_cols=3, figsize=(35, 25))

                    # Define the arrangement of predictor combinations in the grid
                    first_col = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"]["first_col"]
                    second_col = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"]["second_col"]
                    third_col = self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"]["third_col"]

                    # Map predictor combinations to their positions in the grid
                    positions = {}
                    # First column
                    for row, predictor_combination in enumerate(first_col):
                        positions[(row, 0)] = predictor_combination
                    # Second column
                    for row, predictor_combination in enumerate(second_col):
                        positions[(row, 1)] = predictor_combination
                    # Third column (only first row filled)
                    for row, predictor_combination in enumerate(third_col):
                        positions[(row, 2)] = predictor_combination
                    # Empty positions are at (2, 2) and (3, 2)

                    # Define the custom colormap
                    # colors = ["#5E9ACC", "#A3C7A1", "#A3C7A1"]
                    colors = [
                        "#5E9ACC",  # Blue
                        "#4F84B1",  # Deep blue
                        "#9DB9BF",  # Soft gray-blue
                        "#7DA5A9",  # Muted teal
                        "#B9D1B9",  # Light green
                        "#A3C7A1",  # Green
                        "#E3EEE5",  # Very pale green
                        "#CEE7CF",  # Light mint
                    ]

                    # Create the custom colormap
                    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                    feature_combo_name_mapping = self.var_cfg["postprocessing"]["plots"]["feature_combo_name_mapping"]

                    # Iterate over the positions and plot the SHAP beeswarm plots
                    for (row, col), predictor_combination in positions.items():
                        if predictor_combination in data_current:
                            shap_values = data_current[predictor_combination]
                            ax = axes[row][col]
                            # Set the current axis to the subplot
                            plt.sca(ax)
                            # Plot the SHAP beeswarm plot in the specified subplot
                            shap.summary_plot(
                                shap_values.values,
                                shap_values.data,
                                feature_names=shap_values.feature_names,
                                max_display=self.var_cfg["postprocessing"]["plots"]["shap_importance_plot"]["num_to_display"],
                                show=False,
                                plot_size=None,
                                color_bar=False,
                                cmap=cmap
                            )
                            # Set the title of the subplot to the predictor combination name
                            ax.set_title(feature_combo_name_mapping[predictor_combination], fontsize=28, weight="bold")
                            ax.tick_params(axis='both', which='major', labelsize=25)  # Set tick label font size
                            ax.xaxis.label.set_size(25)  # Set x-axis label font size
                            ax.yaxis.label.set_size(25)  # Set y-axis label font size
                        else:
                            print(f"Predictor combination '{predictor_combination}' not found in data.")
                    """
                    # Plot the color bar in the lower right corner (position (3, 2))
                    ax_colorbar = axes[3][2]
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    fig.colorbar(sm, cax=ax_colorbar, orientation='horizontal')
                    ax_colorbar.set_xlabel('Feature value', fontsize=18)
                    ax_colorbar.set_xticks([0, 1])
                    ax_colorbar.set_xticklabels(['Low', 'High'])
                    ax_colorbar.tick_params(axis='both', which='major', labelsize=18)
                    """

                    # Adjust layout and display the figure
                    plt.subplots_adjust(left=0.24, wspace=1.2, hspace=0.4, right=0.95)  # Adjust horizontal and vertical spacing
                    if self.store_plots:
                        self.store_plot(plot_name="beeswarm", crit=crit, samples_to_include=samples_to_include, model=model)
                    else:
                        plt.show()

    def plot_lin_model_coefs(self):
        pass

    def plot_shap_values(self):
        pass

    def plot_shap_ia_values(self):
        pass

    def store_plot(self,
                   plot_name: str,
                   plot_format: str = "png",
                   dpi: int = 450,
                   feature_combination: str = None,
                   samples_to_include: str = None,
                   crit: str = None,
                   model: str = None,
                   ):
        #TODO We could adjust this to a more general method that also stores tables?
        """
        This function is a generic method to store plots in a given directory
            - If we have single plots for specific model/crit/... combinations, we store the plots in the respective dirs
            - If we have a single summarizing plots across everything, we save it in the base folder

        Args:
            plot_name: Name of the plot type, will be used in the filename (e.g. "beeswarm")
            format: Format for the stored image (e.g. "png")
            dpi: Resolution of the plot in dots per inch
            feature_combination: e.g., "pl_srmc"
            samples_to_include: e.g., "all"
            crit: e.g., "state_wb"
            model: e.g., "randomforestregressor"

        """
        plot_path = self.create_plot_path(feature_combination=feature_combination,
                                          samples_to_include=samples_to_include,
                                          crit=crit,
                                          model=model)
        os.makedirs(plot_path, exist_ok=True)
        filename = f"{plot_name}.{plot_format}"
        file_path = os.path.join(plot_path, filename)
        print("store plot in:", file_path)
        plt.savefig(
            file_path, format=plot_format, dpi=dpi,  #...
        )
        plt.close()

    def create_plot_path(self,
                         samples_to_include: str = None,
                         crit: str = None,
                         model: str = None,
                         feature_combination: str = None,
                         ):
        """
        This method creates the path were the plots should be stored

        Args:
            feature_combination:
            samples_to_include:
            crit:
            model:

        Returns:

        """
        path_components = [None, None, None, None]
        for path_idx, var in enumerate([samples_to_include, crit, model, feature_combination]):
            if var is not None:
                path_components[path_idx] = var
                print(path_components)
        filtered_path_components = [comp for comp in path_components if comp]
        return os.path.normpath(os.path.join(self.plot_base_dir, *filtered_path_components))
