import os
from itertools import product
from typing import Callable, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap


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
        self.plot_cfg = self.var_cfg["postprocessing"]["plots"]
        self.plot_base_dir = os.path.join(plot_base_dir, "plots")  # this is the folder containing the processed final results
        self.store_plots = self.plot_cfg["store_plots"]
        self.feature_combo_mapping = self.plot_cfg["feature_combo_name_mapping"]

    def plot_cv_results_plots_wrapper(self, data_to_plot: dict, rel: Union[float, None] = None):
        """
        Wrapper function for the "plot_cv_results_plots" function. It
            - gets the data-indepdenent parameters for the plot from the config
            - iterates over crit - samples_to_include combinations
            - gets the specific data for a given combination
            - invokes the plot_cv_results_plots function for a given combination

        Args:
            data_to_plot (dict): Dict containing the cv_results
            rel: Reliability of the specific crit, if None, it is not included in the plots
        """
        # Get meta-params of the plot that are equal for all combinations. Could be a seperate method though
        color_dct = self.plot_cfg["cv_results_plot"]["color_dct"]
        feature_combinations = []
        for col, feature_combination_lst in self.plot_cfg["cv_results_plot"]["col_assignment"].items():
            feature_combinations.append(feature_combination_lst)

        titles = self.plot_cfg["cv_results_plot"]["titles"]
        models = self.plot_cfg["cv_results_plot"]["models"]
        cv_results_figure_params = self.plot_cfg["cv_results_plot"]["figure"]
        cv_results_empty_cells = [tuple(cell) for cell in cv_results_figure_params['empty_cells']]
        cv_results_fontsizes = self.plot_cfg["cv_results_plot"]["fontsizes"]

        # With this order, we can vary "samples_to_include" in one plot more easily
        for crit in self.plot_cfg["cv_results_plot"]["crit"]:
            for samples_to_include in self.plot_cfg["cv_results_plot"]["samples_to_include"]:
                # Create a new figure for every combination
                fig, axes = self.create_grid(
                    num_cols=cv_results_figure_params["num_cols"],
                    num_rows=cv_results_figure_params["num_rows"],
                    figsize=(cv_results_figure_params["width"], cv_results_figure_params["height"]),
                    empty_cells=cv_results_empty_cells
                )
                # Get the specific data to plot
                filtered_metrics, filtered_metrics_col = self.prepare_cv_results_plot_data(
                    data_to_plot=data_to_plot,
                    crit=crit,
                    samples_to_include=samples_to_include,
                    models=models,
                    feature_combinations=feature_combinations
                )
                # Create margin dct to display the incremental performance
                pl_margin_dct = self.compute_pl_margin(filtered_metrics)

                # Create dict with reference values for the incremental performance
                pl_ref_dct = {
                    "elasticnet": filtered_metrics_col[0]["pl_elasticnet"],
                    "randomforestregressor": filtered_metrics_col[0]["pl_randomforestregressor"]
                }

                self.plot_cv_results_plots(
                    feature_combinations=feature_combinations,
                    crit=crit,
                    samples_to_include=samples_to_include,
                    titles=titles,
                    filtered_metrics_col=filtered_metrics_col,
                    pl_margin_dct=pl_margin_dct,
                    pl_ref_dct=pl_ref_dct,
                    fig=fig,
                    axes=axes,
                    models=models,
                    color_dct=color_dct,
                    fontsizes=cv_results_fontsizes,
                    figure_params=cv_results_figure_params,
                    rel=rel,
                    )

    def prepare_cv_results_plot_data(self, data_to_plot, crit, samples_to_include, models, feature_combinations):
        """
        Prepares the filtered data for CV results plotting.

        Args:
            data_to_plot (dict): The data to be filtered.
            crit (str): The current criterion being processed.
            samples_to_include (str): The sample type to include in the filtering.
            models (list): List of models to include.
            feature_combinations (list of lists): Feature groups to consider for filtering.

        Returns:
            list of dict: Prepared metrics for each feature group.
        """
        filtered_metrics = {}
        filtered_metrics_col = []

        if samples_to_include in ["all", "selected"]:
            # Filter metrics to include only current crit / samples_to_include
            filtered_metrics = {
                key.replace(f"{crit}_", "").replace(f"_{samples_to_include}", ""): value
                for key, value in data_to_plot.items()
                if key.startswith(crit) and key.endswith(samples_to_include)
            }
            # Prepare filtered metrics for each bar to plot
            filtered_metrics_col = [
                {
                    key: value for key, value in filtered_metrics.items()
                    if key in [f"{prefix}_{model}" for prefix, model in product(group, models)]
                }
                for group in feature_combinations
            ]
        elif samples_to_include == "combo":
            for i, group in enumerate(feature_combinations):
                # Use "selected" for the first column (one-level) and "all" for the other coplumns (two/three-level)
                if i == 0:
                    sublist_sample = "selected"
                else:
                    sublist_sample = "all"

                # Filter metrics for the current sublist_sample
                filtered_metrics = {
                    key.replace(f"{crit}_", "").replace(f"_{sublist_sample}", ""): value
                    for key, value in data_to_plot.items()
                    if key.startswith(crit) and key.endswith(sublist_sample)
                }

                # Filter metrics for the current feature group
                group_metrics = {
                    key: value for key, value in filtered_metrics.items()
                    if key in [f"{prefix}_{model}" for prefix, model in product(group, models)]
                }
                filtered_metrics_col.append(group_metrics)
        else:
            raise ValueError("Invalid value for samples_to_include. Must be one of ['all', 'selected', 'combo'].")

        return filtered_metrics, filtered_metrics_col

    def plot_cv_results_plots(self,
                              feature_combinations: list[list[str]],
                              crit: str,
                              samples_to_include: str,
                              titles: list[str],
                              filtered_metrics_col: list[dict],
                              pl_margin_dct: dict,
                              pl_ref_dct: dict,
                              fig,
                              axes,
                              models,
                              color_dct,
                              fontsizes,
                              figure_params,
                              rel=None,
                              ):
        """ # TODO: add significance brackets for both comparison (ENR / RFR as well as incremental change?)
        This function creates the cv_result bar plot for a given analysis (i.e., a samples_to_include / crit combination).
        As the meta-parameters of the plots are equal for all combinations, we pass them as arguments here.

        Args:
            feature_combinations:
            crit:
            samples_to_include:
            titles:
            filtered_metrics_col:
            pl_margin_dct:
            fig:
            axes:
            models:
            color_dct:
            fontsizes:
            figure_params:
            rel:
        """
        # Loop over columns (one-level / two-level / three-level) and rows (different feature combinations)
        for col_idx, (group, title, metrics) in enumerate(zip(feature_combinations, titles, filtered_metrics_col)):
            for row_idx, category in enumerate(group):
                ax = axes[row_idx, col_idx]
                single_category_metrics = {
                    f"{category}_{model}": metrics[f"{category}_{model}"] for model in models if
                    f"{category}_{model}" in metrics
                }
                if col_idx == 0:
                    # Plot for the first column (Within Conceptual Levels)
                    self.plot_bar_plot(
                        ax=ax,
                        data=single_category_metrics,
                        order=[category],
                        models=models,
                        color_dct=color_dct,
                        feature_combo_mapping=self.feature_combo_mapping,
                        rel=rel
                    )
                elif col_idx == 1:
                    # Plot for the second column (Across Two Conceptual Levels)
                    self.plot_incremental_bar_plot(
                        ax=ax,
                        data=single_category_metrics,
                        pl_margin_dct=pl_margin_dct,
                        order=[category],
                        feature_combo_mapping=self.feature_combo_mapping,
                        models=models,
                        color_dct=color_dct,
                        rel=rel,
                        pl_reference_dct=pl_ref_dct
                    )
                elif col_idx == 2 and row_idx < 1:  # < 2:  # TODO Adjust bar color problem
                    # Plot in the third column (Across three conceptual levels, only the first 2 rows)
                    self.plot_incremental_bar_plot(
                        ax=ax,
                        data=single_category_metrics,
                        pl_margin_dct=pl_margin_dct,
                        order=[category],
                        feature_combo_mapping=self.feature_combo_mapping,
                        models=models,
                        color_dct=color_dct,
                        rel=rel,
                        pl_reference_dct=pl_ref_dct
                    )

                ax.set_xlim(figure_params["x_min"], figure_params["x_max"])
                ax.set_title(
                    title if row_idx == 0 else "",
                    fontsize=fontsizes["tick_params"],
                    pad=figure_params["title_pad"],
                    fontweight="bold"
                )
                ax.tick_params(axis='x', labelsize=fontsizes["tick_params"])
                ax.tick_params(axis='y', labelsize=fontsizes["tick_params"])

        # Create a legend in the lower right corner
        randomforest_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='RandomForestRegressor (Upper Bar)')
        elasticnet_patch = mpatches.Patch(facecolor='white', edgecolor='black', label='ElasticNet (Lower Bar)')
        legends_to_handle = [randomforest_patch, elasticnet_patch]
        if rel:
            reliability_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='Reliability = .90')
            legends_to_handle.append(reliability_line)

        # Add the custom legend to the figure
        fig.legend(
            handles=legends_to_handle,
            loc='lower right',
            bbox_to_anchor=(0.98, 0.15),
            ncol=1,
            frameon=False,
            title="",
            fontsize=fontsizes["legend"]
        )

        # Adjust layout for readability
        plt.tight_layout(rect=figure_params["tight_layout"])   # (rect=[0, 0.05, 1, 0.95])
        if self.store_plots:
            self.store_plot(
                plot_name="cv_results",
                plot_format="pdf",  # TODO test
                dpi=600,  # TODO test
                crit=crit,
                samples_to_include=samples_to_include,
                model=None
            )
        else:
            plt.show()

    def plot_bar_plot(self, ax, data, order, models, color_dct, feature_combo_mapping, rel=None):
        """
        Plots a bar plot for the given data with error bars, with each feature group displayed separately.
        Uses different saturation levels to differentiate models and adds a vertical line for `rel`.

        Args:
            ax: matplotlib axis object to plot on.
            data: dict, where keys are feature-model combinations (e.g., 'pl_elasticnet') and values are dictionaries.
            order: list of feature combinations (e.g., ['pl', 'srmc', 'mac', 'sens']) to control y-axis order.
            models: list of models (e.g., ['elasticnet', 'randomforestregressor']) to color the bars.
            color_dct: Optional dict of colors for each feature category.
            rel: vertical line value to indicate a threshold.
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

                # Define the coordinates for the bracket  TODO: Fix or remove
                # left, right = ax.get_xlim()
                # bottom, top = ax.get_ylim()
                # may place some significance bracket here

        # Set y-ticks to the feature group labels in reversed order
        ax.set_yticks(y_positions + bar_width / 2)

        new_lst = [feature_combo_mapping[feature] for feature in order]
        ax.set_yticklabels(new_lst, fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        if "spearman" in metric:
            ax.set_xlabel(r'$\rho$', fontsize=15)
        # ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Draw the rel line if rel is provided
        if rel:
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
                                  rel=None):
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
        # TODO: Add two color option

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
            if rel:
                ax.axvline(x=rel, color='black', linestyle='--', linewidth=1.2, label=f'Rel ({rel:.2f})')

    def add_significance_bracket(self, ax, y1, y2, significance_level):
        """  # TODO Not used ATM
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

    def create_grid(self, num_rows: int, num_cols: int, figsize: tuple[int]=(15, 20), empty_cells=None):
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

    def compute_pl_margin(self, filtered_metrics: dict):
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

        """  # TODO: Remove or include interactions, leave out for now
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
        # We create plot for all combinations
        crits = self.plot_cfg["shap_beeswarm_plot"]["crit"]
        samples_to_include_list = self.plot_cfg["shap_beeswarm_plot"]["samples_to_include"]
        models = self.plot_cfg["shap_beeswarm_plot"]["prediction_model"]

        # Map predictor combinations to their positions in the grid as defined in the config
        col_assignment = self.plot_cfg["shap_beeswarm_plot"]["col_assignment"]
        positions = {}
        for col_idx, (col_name, column) in enumerate(col_assignment.items()):
            for row_idx, predictor_combination in enumerate(column):
                positions[(row_idx, col_idx)] = predictor_combination

        # Create custom colormap
        colors = self.plot_cfg["custom_cmap_colors"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        feature_combo_name_mapping = self.plot_cfg["feature_combo_name_mapping"]
        num_to_display = self.plot_cfg["shap_beeswarm_plot"]["num_to_display"]

        beeswarm_figure_params = self.plot_cfg["shap_beeswarm_plot"]["figure"]
        beeswarm_fontsizes = self.plot_cfg["shap_beeswarm_plot"]["fontsizes"]
        beeswarm_subplot_adj = self.plot_cfg["shap_beeswarm_plot"]["subplot_adjustments"]

        # Iterate over each combination of crit, samples_to_include, and model
        for crit in crits:
            for samples_to_include in samples_to_include_list:
                for model in models:
                    print(f"### Plot combination: {samples_to_include}_{crit}_{model}")
                    data_current = prepare_data_func(
                        crit_to_plot=crit,
                        samples_to_include=samples_to_include,
                        model_to_plot=model,
                        col_assignment=col_assignment,
                    )

                    # Create a grid of subplots with specified empty cells
                    fig, axes = self.create_grid(
                        num_rows=beeswarm_figure_params["num_rows"],
                        num_cols=beeswarm_figure_params["num_cols"],
                        figsize=(beeswarm_figure_params["width"],
                                 beeswarm_figure_params["height"])
                    )

                    # Iterate over the positions and plot the SHAP beeswarm plots
                    for (row, col), predictor_combination in positions.items():
                        if predictor_combination in data_current:
                            shap_values = data_current[predictor_combination]
                            ax = axes[row][col]
                            plt.sca(ax)

                            # Plot the SHAP beeswarm plot in the specified subplot
                            shap.summary_plot(
                                shap_values.values,
                                shap_values.data,
                                feature_names=shap_values.feature_names,
                                max_display=num_to_display,
                                show=False,
                                plot_size=None,
                                color_bar=False,
                                cmap=cmap
                            )
                            # Set title and fontsizes
                            ax.set_title(
                                feature_combo_name_mapping[predictor_combination],
                                fontsize=beeswarm_fontsizes["title"],
                                weight="bold"
                            )
                            ax.tick_params(
                                axis='both',
                                which='major',
                                labelsize=beeswarm_fontsizes["tick_params"]
                            )
                            ax.xaxis.label.set_size(beeswarm_fontsizes["x_label"])
                            ax.yaxis.label.set_size(beeswarm_fontsizes["y_label"])
                        else:
                            print(f"Predictor combination '{predictor_combination}' not found in data.")

                    # Adjust layout and display the figure
                    plt.subplots_adjust(
                        left=beeswarm_subplot_adj["left"],
                        wspace=beeswarm_subplot_adj["wspace"],
                        hspace=beeswarm_subplot_adj["hspace"],
                        right=beeswarm_subplot_adj["right"]
                    )
                    if self.store_plots:
                        self.store_plot(
                            plot_name="beeswarm",
                            crit=crit,
                            samples_to_include=samples_to_include,
                            model=model
                        )
                    else:
                        plt.show()

    def plot_lin_model_coefs(self):
        pass

    def plot_shap_values(self):
        pass

    def plot_shap_ia_values(self):
        pass

    def plot_pred_true_parity(self, sample_data, feature_combination: str, samples_to_include: str, crit: str, model: str):
        """
        This function creates a parity plot of predicted vs. true values for all samples.
        Different samples are plotted with different colors.

        Args:
            sample_data (dict): Dictionary of samples to plot
            feature_combination: e.g., "pl_srmc"
            samples_to_include: e.g., "all"
            crit: e.g., "state_wb"
            model: e.g., "randomforestregressor"
        """

        # After collecting data, create the parity plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Assign colors to samples
        samples = list(sample_data.keys())
        num_samples = len(samples)

        # Get style fitting colors
        colors = self.plot_cfg["custom_cmap_colors"]
        # This works for any number of samples, as it loops through available colors
        colors = [colors[i % len(colors)] for i in range(num_samples)]

        for i, sample_name in enumerate(samples):
            color = colors[i % 10]
            pred_values = sample_data[sample_name]['pred']
            true_values = sample_data[sample_name]['true']
            ax.scatter(true_values, pred_values, color=color, label=sample_name)

        # Plot y = x line for reference
        min_val = min([min(sample_data[s]['true'] + sample_data[s]['pred']) for s in samples])
        max_val = max([max(sample_data[s]['true'] + sample_data[s]['pred']) for s in samples])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')

        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'Pred vs True - {samples_to_include} - {crit} - {model}')
        ax.legend()
        if self.store_plots:
            self.store_plot(plot_name="pred_vs_true_scatter", crit=crit, samples_to_include=samples_to_include, model=model)
        else:
            plt.show()

    def store_plot(self,
                   plot_name: str,
                   plot_format: str = "png",
                   dpi: int = 450,
                   feature_combination: str = None,
                   samples_to_include: str = None,
                   crit: str = None,
                   model: str = None,
                   ):
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
            file_path, format=plot_format, dpi=dpi,
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
            str: path where the plot should be stored

        """
        path_components = [None, None, None, None]
        for path_idx, var in enumerate([samples_to_include, crit, model, feature_combination]):
            if var is not None:
                path_components[path_idx] = var
                print(path_components)
        filtered_path_components = [comp for comp in path_components if comp]

        return os.path.normpath(os.path.join(self.plot_base_dir, *filtered_path_components))
