import os
from itertools import product
from typing import Callable, Union

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FuncFormatter


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

    def plot_cv_results_plots_wrapper(self, data_to_plot: dict, metric: str, rel: Union[float, None] = None):
        """
        Wrapper function for the "plot_cv_results_plots" function. It
            - gets the data-indepdenent parameters for the plot from the config
            - iterates over crit - samples_to_include combinations
            - gets the specific data for a given combination
            - invokes the plot_cv_results_plots function for a given combination

        Args:
            data_to_plot (dict): Dict containing the cv_results
            metric:
            rel: Reliability of the specific crit, if None, it is not included in the plots
        """
        # Get meta-params of the plot that are equal for all combinations. Could be a seperate method though
        color_dct = self.plot_cfg["cv_results_plot"]["color_dct"]
        feature_combinations = []
        for col, feature_combination_lst in self.plot_cfg["cv_results_plot"]["col_assignment"].items():
            feature_combinations.append(feature_combination_lst)

        titles = self.plot_cfg["cv_results_plot"]["titles"]
        models = self.plot_cfg["cv_results_plot"]["models"][::-1]  # so that ENR is displayed above RFR
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
                pl_margin_dct = self.compute_pl_margin(filtered_metrics, metric)

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
                    metric=metric,
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
                              metric,
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
                        row_idx=row_idx,
                        data=single_category_metrics,
                        order=[category],
                        models=models,
                        color_dct=color_dct,
                        feature_combo_mapping=self.feature_combo_mapping,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
                        rel=rel,
                    )
                elif col_idx == 1 or col_idx == 2 and row_idx < 1:
                    # Plot for the second and third column (Across Two Conceptual Levels)
                    self.plot_incremental_bar_plot(
                        ax=ax,
                        row_idx=row_idx,
                        data=single_category_metrics,
                        pl_margin_dct=pl_margin_dct,
                        order=[category],
                        feature_combo_mapping=self.feature_combo_mapping,
                        models=models,
                        color_dct=color_dct,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
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
                # Remove frames around the plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Show x-axis only for the bottom row
                if row_idx < len(group) - 1:
                    ax.tick_params(axis='x', bottom=False, labelbottom=False)
                    ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='y', labelsize=fontsizes["tick_params"])

        # Create a legend in the lower right corner
        elasticnet_patch = mpatches.Patch(facecolor='lightgray', edgecolor='black', label='ENR (Upper Bar)')
        randomforest_patch = mpatches.Patch(facecolor='gray', edgecolor='black', label='RFR (Lower Bar)')
        personal_patch = mpatches.Patch(facecolor=color_dct["pl"], edgecolor='none', label='Personal Predictors')
        situational_patch = mpatches.Patch(facecolor=color_dct["srmc"], edgecolor='none', label='Situational/Societal Predictors')
        legends_to_handle = [elasticnet_patch, randomforest_patch, personal_patch, situational_patch]
        if rel:
            reliability_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='Reliability = .90')
            legends_to_handle.append(reliability_line)

        # Add the custom legend to the figure
        fig.legend(
            handles=legends_to_handle,
            loc='lower right',
            bbox_to_anchor=tuple(figure_params["legend_pos"]),
            ncol=1,
            frameon=False,
            title="",
            fontsize=fontsizes["legend"]
        )
        #line_x_positions = [0.1235, 0.437, 0.755]  # adjust these as needed
        #line_y_positions = [[0.1, 0.9], [0.1, 0.9], [0.52, 0.9]]
        line_x_positions = [0.09, 0.41, 0.728]
        line_y_positions = [[0.1, 0.9], [0.1, 0.9], [0.52, 0.9]]

        for x, y in zip(line_x_positions, line_y_positions):
            line = mlines.Line2D(
                xdata=[x, x],
                ydata=y,
                transform=fig.transFigure,
                color='black',
                linestyle='-',
                linewidth=1
            )
            fig.add_artist(line)

        # Adjust layout for readability
        plt.tight_layout(rect=figure_params["tight_layout"])  # (rect=[0, 0.05, 1, 0.95])
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

    def plot_bar_plot(self,
                      ax,
                      data,
                      order,
                      models,
                      color_dct,
                      feature_combo_mapping,
                      fontsizes,
                      metric,
                      bar_width,
                      row_idx,
                      rel=None):
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
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"

        # Generate bar positions for each feature group
        y_positions = np.arange(len(order))

        # Loop through each model and feature group
        for i, model in enumerate(models):
            for j, feature in enumerate(order):
                # Set color saturation to differentiate models
                base_color = color_dct[feature]
                alpha = 1 if model == "randomforestregressor" else 0.7

                model_key = f"{feature}_{model}"
                value = data.get(model_key, {}).get(m_metric, 0)
                error = data.get(model_key, {}).get(sd_metric, 0)

                # Plot the bar with adjusted color
                ax.barh(y_positions[j] + (i * bar_width), value, xerr=error, height=bar_width,
                        color=base_color, align='center', capsize=5, edgecolor=None, alpha=alpha)

                # Define the coordinates for the bracket  TODO: Fix or remove
                # left, right = ax.get_xlim()
                # bottom, top = ax.get_ylim()
                # may place some significance bracket here

        self.format_bar_plot(
            ax=ax,
            row_idx=row_idx,
            y_positions=y_positions,
            bar_width=bar_width,
            feature_combo_mapping=feature_combo_mapping,
            order=order,
            metric=metric,
            fontsizes=fontsizes,
            rel=rel
        )

    def plot_incremental_bar_plot(self,
                                  ax,
                                  data,
                                  pl_margin_dct,
                                  pl_reference_dct,
                                  order,
                                  models,
                                  color_dct,
                                  feature_combo_mapping,
                                  fontsizes,
                                  metric,
                                  row_idx,
                                  bar_width,
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
        print(metric)
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"
        y_positions = np.arange(len(order))

        for i, model in enumerate(models):
            for j, feature in enumerate(order):
                # Combination, plot incremental bar
                base_value = pl_reference_dct[model][m_metric]
                increment = pl_margin_dct.get(f"{feature}_{model}", {}).get(f"incremental_{m_metric}", 0)
                error = data[f"{feature}_{model}"][sd_metric]

                base_feature = feature.split("_")[1]
                other_feat_color = color_dct[base_feature]
                pl_color = color_dct["pl"]
                alpha = 1 if model == "randomforestregressor" else 0.7

                ax.barh(y_positions[j] + (i * bar_width), base_value, height=bar_width, color=pl_color,
                        align='center', edgecolor=None, alpha=alpha)
                if increment > 0:
                    ax.barh(y_positions[j] + (i * bar_width), increment, left=base_value, height=bar_width,
                            xerr=error, color=other_feat_color, align='center', edgecolor=None, alpha=alpha,
                            capsize=5)
                else: # if negative, just show the pl color and dislplay true results
                    ax.barh(y_positions[j] + (i * bar_width), 0, left=base_value, height=bar_width,
                            xerr=error, color=other_feat_color, align='center', edgecolor=None, alpha=alpha,  # hatch
                            capsize=5)

            self.format_bar_plot(
                ax=ax,
                row_idx=row_idx,
                y_positions=y_positions,
                bar_width=bar_width,
                feature_combo_mapping=feature_combo_mapping,
                order=order,
                metric=metric,
                fontsizes=fontsizes,
                rel=rel
            )

    def format_bar_plot(self, ax, row_idx, y_positions, bar_width, feature_combo_mapping, order, fontsizes, metric, rel=None):
        """
        Formats the bar plot by setting y-ticks, x-axis label based on the metric, and an optional reference line.

        Args:
            ax: Matplotlib Axes object to format.
            y_positions: Array of y positions for the bars.
            bar_width: Width of the bars.
            feature_combo_mapping: Dictionary mapping features to their group labels.
            order: List of features in the desired order for the y-axis.
            metric: Metric for the x-axis label ("spearman", "pearson", or "r2").
            rel: Optional float, the reference line value to draw.
        """
        # Set y-ticks to the feature group labels in reversed order
        ax.set_yticks(y_positions + bar_width / 2)

        # Map features to their group labels
        feature_combos = [feature_combo_mapping[feature] for feature in order]
        # Format feature combinations
        feature_combos_str_format = [self.line_break_strings(combo, max_char_on_line=14, balance=False)
                                     for combo in feature_combos]
        ax.set_yticklabels(feature_combos_str_format,
                           fontsize=fontsizes["tick_params"],
                           horizontalalignment="left")

        ax.spines['left'].set_visible(False)

        # Apply custom APA-style formatting to the x-axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: self.format_metric(x)))

        # Set tick parameters
        ax.tick_params(axis='x', labelsize=fontsizes["tick_params"])
        ax.tick_params(axis='y', which='major', pad=90, length=0)

        # Set x-axis label based on the metric
        if row_idx == 3:
            if metric == "spearman":
                ax.set_xlabel(r'$\rho$', fontsize=fontsizes["label"], labelpad=10)
            elif metric == "pearson":
                ax.set_xlabel(r'$r$', fontsize=fontsizes["label"], labelpad=10)  # Italicized r in LaTeX
            elif metric == "r2":
                ax.set_xlabel(r'$R^2$', fontsize=fontsizes["label"], labelpad=10)

        # Draw the reference line if provided
        if rel is not None:
            ax.axvline(x=rel, color='black', linestyle='--', linewidth=1.2, label=f'Rel ({rel:.2f})')

    @staticmethod
    def format_metric(metric: float) -> str:
        """
        Formats a given metric according to APA style:
        - No leading zero for correlations or R² values (e.g., .85 instead of 0.85).
        - Rounded to two decimal places.

        Args:
            metric: The metric to be formatted (e.g., a correlation or R² value).

        Returns:
            A string representation of the formatted metric.
        """
        if metric == 0:
            return 0

        if not isinstance(metric, (float, int)):
            raise ValueError("The metric must be a numeric value.")

        # Round to two decimals
        rounded_metric = round(metric, 2)

        # Format without leading zero
        formatted_metric = f"{rounded_metric:.2f}".lstrip('0').replace('-.', '-0.')

        return formatted_metric

    def apply_formatted_x_axis(self, ax):
        """
        Applies the APA-style formatting to the x-axis of a plot.

        Args:
            ax: The Matplotlib Axes object whose x-axis needs formatting.
        """
        formatter = FuncFormatter(lambda x, _: self.format_metric(x))
        ax.xaxis.set_major_formatter(formatter)

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

    def create_grid(self, num_rows: int, num_cols: int, figsize: tuple[int] = (15, 20), empty_cells=None):
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

    def compute_pl_margin(self, filtered_metrics: dict, metric: str):
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
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"

        # Get base metrics for 'pl' only (e.g., 'pl_elasticnet' and 'pl_randomforestregressor')
        base_metrics = {key: value for key, value in filtered_metrics.items() if key.startswith("pl_") and "_" not in key[len("pl_"):]}

        # Loop over each key in the filtered_metrics that contains 'pl_' but is not the base 'pl'
        for key, value in filtered_metrics.items():
            if key.startswith("pl_") and key not in base_metrics:
                model = key.split('_')[-1]
                base_key = f"pl_{model}"

                # Check if the base metric exists
                if base_key in base_metrics:
                    # Calculate the difference in metric  for the combination vs the base 'pl'
                    incremental_difference = value[m_metric] - base_metrics[base_key][m_metric]

                    # Store the incremental difference with the key indicating the combination and model
                    pl_margin_dict[key] = {
                        f'incremental_m_{metric}': incremental_difference,
                        sd_metric: value[sd_metric]  # Keep the standard deviation of the combination
                    }

        return pl_margin_dict

    def plot_shap_beeswarm_plots(self, prepare_shap_data_func: Callable, prepare_shap_ia_data_func: Callable = None):
        """
        Plots SHAP beeswarm plots for different predictor combinations arranged in a grid.

        Args:
            prepare_shap_data_func
            prepare_shap_ia_data_func

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
        ia_data_current = None

        # Iterate over each combination of crit, samples_to_include, and model
        for crit in crits:
            for samples_to_include in samples_to_include_list:
                for model in models:
                    print(f"### Plot combination: {samples_to_include}_{crit}_{model}")
                    data_current = prepare_shap_data_func(
                        crit_to_plot=crit,
                        samples_to_include=samples_to_include,
                        model_to_plot=model,
                        col_assignment=col_assignment,
                    )
                    if self.plot_cfg["shap_beeswarm_plot"]["shap_ia_values"]["add"] and model == "randomforestregressor":
                        # Get IA data from other function
                        ia_cfg = self.plot_cfg["shap_beeswarm_plot"]["shap_ia_values"]
                        print("Now get ia values")
                        ia_data_current = prepare_shap_ia_data_func(
                            crit_to_plot=crit,
                            samples_to_include=samples_to_include,
                            model_to_plot=model,
                            feature_combination=ia_cfg["feature_combination"],
                            meta_stat_to_extract=ia_cfg["meta_stat_to_extract"],
                            stat_to_extract=ia_cfg["stat_to_extract"],
                            order_to_extract=ia_cfg["order_to_extract"],
                            num_to_extract=ia_cfg["num_to_extract"]
                        )
                        ia_position = tuple(ia_cfg["position"])
                        positions[ia_position] = next(iter(ia_data_current))

                    # Create a grid of subplots with specified empty cells
                    fig, axes = self.create_grid(
                        num_rows=beeswarm_figure_params["num_rows"],
                        num_cols=beeswarm_figure_params["num_cols"],
                        figsize=(int(beeswarm_figure_params["width"]),
                                 int(beeswarm_figure_params["height"]))
                    )

                    # Iterate over the positions and plot the SHAP beeswarm plots
                    for (row, col), predictor_combination in positions.items():
                        if predictor_combination in data_current:
                            shap_values = data_current[predictor_combination]
                            self.plot_shap_beeswarm(
                                shap_values=shap_values,
                                ax=axes[row, col],
                                beeswarm_fontsizes=beeswarm_fontsizes,
                                num_to_display=num_to_display,
                                cmap=cmap,
                                feature_combo_name_mapping=feature_combo_name_mapping,
                                predictor_combination=predictor_combination,
                            )
                        else:
                            if predictor_combination in ia_data_current:
                                shap_ia_data = ia_data_current[predictor_combination]
                                self.plot_shap_beeswarm(
                                    shap_values=shap_ia_data,
                                    ax=axes[row, col],
                                    beeswarm_fontsizes=beeswarm_fontsizes,
                                    num_to_display=num_to_display,
                                    cmap=cmap,
                                    feature_combo_name_mapping=feature_combo_name_mapping,
                                    predictor_combination=predictor_combination,
                                )
                            else:
                                print(f"Predictor combination '{predictor_combination}' not found in shap data or shap ia data")

                    # Adjust layout and display the figure
                    plt.subplots_adjust(
                        left=beeswarm_subplot_adj["left"],
                        wspace=beeswarm_subplot_adj["wspace"],
                        hspace=beeswarm_subplot_adj["hspace"],
                        right=beeswarm_subplot_adj["right"]
                    )
                    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.975])
                    if self.store_plots:
                        self.store_plot(
                            plot_name=f"beeswarm_{crit}_{samples_to_include}_{model}",
                            crit=crit,
                            samples_to_include=samples_to_include,
                            plot_format="pdf",
                            model=model
                        )
                    else:
                        plt.show()

    def plot_shap_beeswarm(self,
                           shap_values: shap.Explanation,
                           ax,
                           beeswarm_fontsizes: dict,
                           num_to_display: int,
                           cmap,
                           feature_combo_name_mapping,
                           predictor_combination,
                           ia_values: bool = False,
                           ):
        """

        Returns:

        """
        plt.sca(ax)
        feature_names_formatted = [self.line_break_strings(feature_name, max_char_on_line=28, balance=True)
                                   for feature_name in shap_values.feature_names]

        # Plot the SHAP beeswarm plot in the specified subplot
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=feature_names_formatted,
            max_display=num_to_display,
            show=False,
            plot_size=None,
            color_bar=False,
            cmap=cmap,
        )
        # Set title and fontsizes
        if ia_values:
            split_strng = "-"
        else:
            split_strng = "+"

        formatted_title = self.line_break_strings(
            strng=feature_combo_name_mapping[predictor_combination],
            max_char_on_line=26,
            split_strng=split_strng,
        )
        ax.set_title(
            formatted_title,
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

        if ia_values:
            plt.set_cmap(cmap)

    def plot_shap_importances(self,
                              shap_ia_data: dict,
                              ax,
                              beeswarm_fontsizes: dict,
                              num_to_display: int,
                              cmap,
                              feature_combo_name_mapping,
                              predictor_combination,
                              ):
        """ # TODO: Idee: Farbe nach Vorzeichen (positive / negative Interactions)?
        Plot a bar chart of SHAP-like feature importances.

        Args:
            shap_ia_data (dict): A dictionary of feature importances keyed by feature name.
            ax (matplotlib.axes._axes.Axes): The axes object on which to plot.
            beeswarm_fontsizes (dict): Dictionary specifying font sizes for title, tick_params, x_label, and y_label.
            num_to_display (int): The number of top features to display.
            cmap (matplotlib.colors.ListedColormap or str): Colormap for coloring the bars (fallback if feature not in color_dct).
            feature_combo_name_mapping (dict): Dictionary mapping predictor combinations to a formatted string for the title.
            predictor_combination (hashable): Key used to retrieve the appropriate title from `feature_combo_name_mapping`.
            color_dct (dict): Dictionary mapping feature names to specific colors.

        Returns:
            None
        """
        plt.sca(ax)

        # Extract feature names and values
        features = list(shap_ia_data.keys())
        values = [val["mean"] for val in list(shap_ia_data.values())]

        # Limit the number of features displayed
        if num_to_display < len(features):
            features = features[:num_to_display]
            values = values[:num_to_display]

        # Format feature names
        feature_names_formatted = [
            self.line_break_strings(feature_name, max_char_on_line=33, balance=True)
            for feature_name in features
        ]

        # Use provided colors from color_dct, fallback to a color from cmap if needed
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        # Create positions for bars to increase space between them
        # By spacing out the y positions, we can create more distance between bars
        y_positions = range(len(feature_names_formatted))
        bar_height = 0.6  # Reduce bar height to create more space

        # Create the bar plot with specified height for spacing
        ax.barh(y=y_positions, width=values, color="blue", edgecolor='black', height=bar_height)

        # Invert y-axis for SHAP-like order
        ax.invert_yaxis()

        # Set custom y-ticks and labels
        # We'll keep the labels, but remove tick marks. Tick labels help identify the features.
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_names_formatted, fontsize=beeswarm_fontsizes["tick_params"])

        # Remove the y-tick lines but keep the labels
        ax.tick_params(axis='y', which='both', length=0)

        # Show the y-axis line (left spine)
        ax.spines['left'].set_visible(True)

        # Keep the x-axis and label it
        ax.set_xlabel("mean(|SHAP interaction value|)", fontsize=beeswarm_fontsizes["x_label"])
        ax.xaxis.label.set_size(beeswarm_fontsizes["x_label"])

        # Set title
        formatted_title = self.line_break_strings(
            strng="Societal - SHAP Interacion Values",
            max_char_on_line=26,
            split_strng="-"
        )
        ax.set_title(formatted_title, fontsize=beeswarm_fontsizes["title"], weight="bold")

        # Adjust tick label size
        ax.tick_params(axis='x', which='major', labelsize=beeswarm_fontsizes["tick_params"])

        # Remove unnecessary spines except the left one (y-axis line)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)  # Keep bottom spine (x-axis)
        # Left spine is already visible for y-axis

        # Optional grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)

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

    def line_break_strings(self, strng: str, max_char_on_line: int, split_strng: str = None, balance: bool = False) -> str:
        if len(strng) <= max_char_on_line:
            return strng

        substring = strng[:max_char_on_line + 1]

        # Try to break at split_strng if provided
        if split_strng is not None:
            split_pos = substring.rfind(split_strng)
            if split_pos != -1:
                break_pos = split_pos + len(split_strng)
            else:
                break_pos = substring.rfind(' ')
                if break_pos == -1:
                    break_pos = max_char_on_line
        else:
            break_pos = substring.rfind(' ')
            if break_pos == -1:
                break_pos = max_char_on_line

        # If balancing is requested and we have a large discrepancy, try to find a better break
        if balance:
            # Current first cut
            first_line = strng[:break_pos].rstrip()
            remainder = strng[break_pos:].lstrip()

            # Only attempt balancing if there's actually a remainder to split
            if remainder:
                # Measure current imbalance
                current_diff = abs(len(first_line) - len(remainder))

                # Look around the current break_pos for a better space that reduces the difference
                # We'll search both directions near break_pos to find a better balance
                best_pos = break_pos
                best_diff = current_diff

                # Define a search range around break_pos (e.g., ±10 characters) for a better space
                search_range = 10
                start_search = max(break_pos - search_range, 1)
                end_search = min(break_pos + search_range, len(strng) - 1)

                for candidate_pos in range(start_search, end_search + 1):
                    if candidate_pos != break_pos and candidate_pos < len(strng):
                        # Check if this is a viable space or split_strng location
                        # We'll still follow the same "break at space or exact" logic
                        if strng[candidate_pos] == ' ' or (
                                split_strng and strng[candidate_pos - len(split_strng) + 1:candidate_pos + 1] == split_strng):
                            # Test this candidate
                            test_first = strng[:candidate_pos].rstrip()
                            test_rem = strng[candidate_pos:].lstrip()
                            new_diff = abs(len(test_first) - len(test_rem))
                            if new_diff < best_diff:
                                best_diff = new_diff
                                best_pos = candidate_pos

                break_pos = best_pos

        # After potentially adjusting break_pos for balance
        first_line = strng[:break_pos].rstrip()
        remainder = strng[break_pos:].lstrip()

        return first_line + '\n' + self.line_break_strings(remainder, max_char_on_line, split_strng, balance)

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
