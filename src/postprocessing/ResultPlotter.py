import os
import re
from itertools import product
from typing import Callable, Union, Collection

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
from shap import Explanation


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
        # self.feature_combo_mapping = self.plot_cfg["feature_combo_name_mapping"]  # TODO adjust
        self.feature_combo_mapping_cv_result_plot = self.plot_cfg["cv_results_plot"]["feature_combo_name_mapping"]

    @property
    def cv_plot_params(self):
        """
        This property extracts the changeable parameters for the cv_results plot from the config and stores them in a
        dict as a class attribute.

        Returns:
            dict: Containing the config parameters for the cv_results plot.

        """
        return {"colors": self.plot_cfg["cv_results_plot"]["color_dct"],
                "figure_params": self.plot_cfg["cv_results_plot"]["figure_params"],
                "titles": self.plot_cfg["cv_results_plot"]["titles"],
                "models": self.plot_cfg["cv_results_plot"]["models"][::-1],
                "fontsizes": self.plot_cfg["cv_results_plot"]["fontsizes"],
                "feature_combos": [feature_combination_lst for feature_combination_lst
                                   in self.plot_cfg["cv_results_plot"]["col_assignment"].values()],
                "feature_combo_mapping": self.plot_cfg["cv_results_plot"]["feature_combo_name_mapping"],
                "metrics_to_plot": self.plot_cfg["cv_results_plot"]["metrics"],
                "crits_to_plot": self.plot_cfg["cv_results_plot"]["crits"],
                "samples_to_include_to_plot": self.plot_cfg["cv_results_plot"]["samples_to_include"],
                "store_params": self.plot_cfg["cv_results_plot"]["store_params"],
                }

    @property
    def shap_plot_params(self):
        """
        This property extracts the changeable parameters for the cv_results plot from the config and stores them in a
        dict as a class attribute.
        """
        return {}

    def plot_cv_results_plots_wrapper(
                                      self,
                                      cv_results_dct: dict[str, dict[str, dict[str, float]]],
                                      rel: Union[float, None] = None) -> None:
        """
        Wrapper function for the "plot_cv_results_plots" function. It
            - iterates over metric - crit - samples_to_include combinations and creates a new base plot for each
            - gets the specific data for a given combination (including increments and base values for comparison)
            - invokes the plot_cv_results_plots function for a given combination

        Args:
            cv_results_dct: Dict containing the cv_results (m/sd) for a given crit/feature_combo/samples_to_include/model
                combination for each metric
            rel: Reliability of the specific crit, if None, it is not included in the plots
        """
        for metric in self.cv_plot_params["metrics_to_plot"]:
            for crit in self.cv_plot_params["crits_to_plot"]:
                for samples_to_include in self.cv_plot_params["samples_to_include_to_plot"]:
                    fig, axes = self.create_grid(
                        num_cols=self.cv_plot_params["figure_params"]["num_cols"],
                        num_rows=self.cv_plot_params["figure_params"]["num_rows"],
                        figsize=(self.cv_plot_params["figure_params"]["width"],
                                 self.cv_plot_params["figure_params"]["height"]),
                        empty_cells=[tuple(cell) for cell in self.cv_plot_params["figure_params"]['empty_cells']]
                    )

                    # Get the specific data to plot and extract the increment and base values for 2- and 3-level analyses
                    filtered_metrics_col = self.prepare_cv_results_plot_data(
                        cv_results_dct=cv_results_dct[metric],
                        crit=crit,
                        samples_to_include=samples_to_include,
                        models=self.cv_plot_params["models"],
                        feature_combinations=self.cv_plot_params["feature_combos"]
                    )
                    ref_dct = self.get_refs(
                        cv_results_dct=cv_results_dct[metric],
                        crit=crit,
                        samples_to_include="all",
                        ref_feature_combo="pl"
                    )
                    margin_dct = self.get_margins(
                        cv_results_dct=cv_results_dct[metric],
                        ref_dct=ref_dct,
                        metric=metric,
                        crit=crit,
                        samples_to_include="all",
                        ref_feature_combo="pl"
                    )

                    # Create plots for a given metric - crit - samples_to_include combination
                    self.plot_cv_results_plots(
                        feature_combinations=self.cv_plot_params["feature_combos"],
                        crit=crit,
                        samples_to_include=samples_to_include,
                        titles=self.cv_plot_params["titles"],
                        filtered_metrics_col=filtered_metrics_col,
                        margin_dct=margin_dct,
                        ref_dct=ref_dct,
                        fig=fig,
                        axes=axes,
                        models=self.cv_plot_params["models"],
                        color_dct=self.cv_plot_params["colors"],
                        fontsizes=self.cv_plot_params["fontsizes"],
                        figure_params=self.cv_plot_params["figure_params"],
                        metric=metric,
                        rel=rel,
                    )

    def prepare_cv_results_plot_data(self,
                                     cv_results_dct: dict[str, dict[str, float]],
                                     crit: str,
                                     samples_to_include: str,
                                     models: list[str],
                                     feature_combinations: list[list[str]]) \
            -> list[dict[str, dict[str, float]]]:
        """
        Prepares the filtered data for CV results plotting.
            - Filters the data to include only the current crit / samples_to_include
            - Extracts the right metrics for each subplot location on the base plot

        Note:
            For the "combo" scenario:
            - The results for samples_to_include == "selected" are used in the first column (one-level analysis)
            - The results for samples_to_include == "all" are used in the 2nd and 3rd column (two- and three-level analysis).

        Args:
            cv_results_dct: Dict containing the cv_results (m/sd) for a given crit/feature_combo/samples_to_include/model
                and the metric specified
            crit: The current criterion being processed.
            samples_to_include: The sample type to include ("all", "selected", or "combo").
            models: List of models to include.
            feature_combinations: Feature groups to consider for filtering.

        Returns:
            tuple[dict, list[dict]]: (filtered_metrics, filtered_metrics_col)
        """
        if samples_to_include not in ["all", "selected", "combo"]:
            raise ValueError("Invalid value for samples_to_include. Must be one of ['all', 'selected', 'combo'].")

        filtered_metrics_col = []

        for i, group in enumerate(feature_combinations):
            if samples_to_include == "combo":
                sublist_sample = "selected" if i == 0 else "all"
            else:
                sublist_sample = samples_to_include

            filtered_metrics = self.filter_cv_results_data(
                cv_results_dct=cv_results_dct,
                crit=crit,
                samples_to_include=sublist_sample
            )
            filtered_metric_col = {
                key: value for key, value in filtered_metrics.items()
                if key in [f"{prefix}_{model}" for prefix, model in product(group, models)]
            }
            filtered_metrics_col.append(filtered_metric_col)

        return filtered_metrics_col

    @staticmethod
    def filter_cv_results_data(
                               cv_results_dct: dict[str, dict[str, float]],
                               crit: str,
                               samples_to_include: str) -> dict[str, dict[str, float]]:
        """
        This function filters the cv_results_dct for a specific crit and samples_to_include

        Args:
            cv_results_dct:
            crit:
            samples_to_include:

        Returns:
            dict: Containing the filtered dict with adjusted keys

        """
        return {
                key.replace(f"{crit}_", "").replace(f"_{samples_to_include}", ""): value
                for key, value in cv_results_dct.items()
                if key.startswith(crit) and key.endswith(samples_to_include)
            }

    def plot_cv_results_plots(self,
                              feature_combinations: list[list[str]],
                              crit: str,
                              samples_to_include: str,
                              titles: list[str],
                              filtered_metrics_col: list[dict[str, dict[str, float]]],
                              margin_dct: dict[str, dict[str, float]],
                              ref_dct: dict[str, dict[str, float]],
                              fig: Figure,
                              axes: ndarray[Axes],
                              models: list[str],
                              color_dct: dict[str, Union[str, dict[str, float]]],  # dict[str, Union[str, dict[str, [float]]]],
                              fontsizes: dict[str, int],
                              figure_params: dict[str, Union[int, float, list[int], list[float]]],
                              metric: str,
                              rel: float = None,
                              ) -> None:
        """
        This function creates the cv_result bar plot for a given analysis (i.e., a samples_to_include / crit combination).
        As the meta-parameters of the plots are equal for all combinations, we pass them as arguments here.

        Args:
            feature_combinations:
            crit:
            samples_to_include:
            titles:
            filtered_metrics_col:
            margin_dct:
            ref_dct
            fig:
            axes:
            models:
            color_dct:
            fontsizes:
            figure_params:
            metric:
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

                if col_idx == 0:  # Plots of the first column (1-level predictions)
                    self.plot_bar_plot(
                        ax=ax,
                        row_idx=row_idx,
                        data=single_category_metrics,
                        order=[category],
                        models=models,
                        color_dct=color_dct,
                        feature_combo_mapping=self.feature_combo_mapping_cv_result_plot,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
                        rel=rel,
                    )

                elif col_idx == 1 or col_idx == 2 and row_idx < 2:  # Plots of the 2nd and 3rd columns (2- and 3-level predictions)
                    self.plot_incremental_bar_plot(
                        ax=ax,
                        row_idx=row_idx,
                        data=single_category_metrics,
                        pl_margin_dct=margin_dct,
                        order=[category],
                        feature_combo_mapping=self.feature_combo_mapping_cv_result_plot,
                        models=models,
                        color_dct=color_dct,
                        metric=metric,
                        fontsizes=fontsizes,
                        bar_width=figure_params["bar_width"],
                        rel=rel,
                        pl_reference_dct=ref_dct
                    )

                # Set global title, axis labels, and formatting
                ax.set_xlim(figure_params["x_min"], figure_params["x_max"])
                ax.set_title(
                    title if row_idx == 0 else "",
                    fontsize=fontsizes["tick_params"],
                    pad=figure_params["title_pad"],
                    fontweight="bold",
                    loc="center"
                )
                ax.title.set_position(figure_params["title_pos"])
                ax.set_xlim(figure_params["x_min"], figure_params["x_max"])
                ax.tick_params(axis='y', labelsize=fontsizes["tick_params"])

                # Remove frames around the plot and show x-axis only for the bottom row
                for spine in ['top', 'right', 'left']:
                    ax.spines[spine].set_visible(False)
                if row_idx < len(group) - 1:
                    ax.tick_params(axis='x', bottom=False, labelbottom=False)
                    ax.spines['bottom'].set_visible(False)

        self.add_cv_results_plot_legend(
            fig=fig,
            legend_loc=figure_params["legend_pos"],
            legend_fontsize=fontsizes["legend"],
            color_dct=color_dct,
            model_legend=True,
            feature_combo_legend=True,
            rel=False
        )

        # Add a joint y-axis across all subplots
        for x, y in zip(figure_params["y_axis_x_pos"], figure_params["y_axis_y_pos"]):
            line = mlines.Line2D(
                xdata=[x, x],
                ydata=y,
                transform=fig.transFigure,
                color='black',
                linestyle='-',
                linewidth=1,
            )
            fig.add_artist(line)

        # Adjust layout and store
        plt.tight_layout(rect=figure_params["tight_layout"])
        if self.store_plots:
            self.store_plot(
                plot_name=self.cv_plot_params["store_params"]["name"],
                plot_format=self.cv_plot_params["store_params"]["format"],
                dpi=self.cv_plot_params["store_params"]["dpi"],
                crit=crit,
                samples_to_include=samples_to_include,
                model=None
            )
        else:
            plt.show()

    @staticmethod
    def add_cv_results_plot_legend(fig: Figure,
                                   legend_loc: tuple[float, float],
                                   legend_fontsize: int,
                                   color_dct: dict,
                                   model_legend: bool = True,
                                   feature_combo_legend: bool = True,
                                   rel: bool = False) -> None:
        """
        This function adds a legend in the lower right corner of the cv_results plot. This includes
            - a description of the bar location for the models
            - a description of the colors for the feature groups

        Returns:
        """
        legends_to_plot = []
        if model_legend:
            elasticnet_patch = mpatches.Patch(
                facecolor='lightgray',
                edgecolor='none',
                label='ENR (Upper Bar)'
            )
            randomforest_patch = mpatches.Patch(
                facecolor='gray',
                edgecolor='none',
                label='RFR (Lower Bar)'
            )
            legends_to_plot.extend([elasticnet_patch, randomforest_patch])
        if feature_combo_legend:
            personal_patch = mpatches.Patch(
                facecolor=color_dct["pl"],
                edgecolor='none',
                label='Personal Predictors'
            )
            other_patch = mpatches.Patch(
                facecolor=color_dct["other"],
                edgecolor='none',
                label='Situational/Societal Predictors'
            )
            legends_to_plot.extend([personal_patch, other_patch])

        if rel:
            reliability_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label=f'Reliability = {rel}')
            legends_to_plot.append(reliability_line)

        # Add the custom legend to the figure
        fig.legend(
            handles=legends_to_plot,
            loc='lower right',
            bbox_to_anchor=legend_loc,  # tuple(figure_params["legend_pos"]),
            ncol=1,
            frameon=False,
            title="",
            fontsize=legend_fontsize
        )

    def plot_bar_plot(self,
                      ax: Axes,
                      data: dict[str, dict[str, float]],
                      order: list[str],
                      models: list[str],
                      color_dct: dict[str, Union[str, dict[str, float]]],
                      feature_combo_mapping: dict[str, str],
                      fontsizes: dict[str, int],
                      metric: str,
                      bar_width: float,
                      row_idx: int,
                      rel: float = None) -> None:
        """
        Plots a bar plot for the given data with error bars, with each feature group displayed separately.
        Uses different saturation levels to differentiate models and adds a vertical line for `rel`, if provided

        Args:
            ax: matplotlib axis object to plot on.
            data: dict, where keys are feature-model combinations (e.g., 'pl_elasticnet') and values are dictionaries.
            order: list of feature combinations (e.g., ['pl', 'srmc', 'mac', 'sens']) to control y-axis order.
            models: list of models (e.g., ['elasticnet', 'randomforestregressor']) to color the bars.
            color_dct: Optional dict of colors for each feature category.
            feature_combo_mapping:
            fontsizes
            metric
            bar_width
            row_idx
            rel: vertical line value to indicate a threshold.
        """
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"
        y_positions = np.arange(len(order))

        # Loop through each model and feature group
        for i, model in enumerate(models):
            for j, feature in enumerate(order):
                # Set color saturation to differentiate models
                if feature == "pl":
                    base_color = color_dct["pl"]
                else:
                    base_color = color_dct["other"]
                alpha = color_dct["bar_saturation"][model]

                model_key = f"{feature}_{model}"
                value = data[model_key][m_metric]
                error = data[model_key][sd_metric]

                # Plot the bar with adjusted color
                ax.barh(y_positions[j] + (i * bar_width), value, xerr=error, height=bar_width,
                        color=base_color, align='center', capsize=5, edgecolor=None, alpha=alpha)

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
                                  ax: Axes,
                                  data: dict[str, dict[str, float]],
                                  pl_margin_dct: dict[str, dict[str, float]],
                                  pl_reference_dct: dict[str, dict[str, float]],
                                  order: list[str],
                                  models: list[str],
                                  color_dct: dict[str, Union[str, dict[str, float]]],
                                  feature_combo_mapping: dict[str, str],
                                  fontsizes: dict[str, int],
                                  metric: str,
                                  row_idx: int,
                                  bar_width: float,
                                  rel: float = None) -> None:
        """
        Plots an incremental bar plot, splitting each bar into 'pl' base and the incremental effect.

        Args:
            ax: matplotlib axis object to plot on.
            data: dict of feature-model combinations with metrics.
            pl_margin_dct: dict from get_margins, containing incremental metric values
            pl_reference_dct: dict from get_refs, containing base metric values
            order: list of feature combinations.
            models: list of models.
            color_dct: Color dictionary for each feature.
            feature_combo_mapping:
            fontsizes
            metric
            bar_width
            row_idx
            rel: vertical line value to indicate a threshold.
        """
        print(metric)
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"
        y_positions = np.arange(len(order))

        for i, model in enumerate(models):
            for j, feature in enumerate(order):

                # Combination, plot incremental bar
                base_value = pl_reference_dct[f"pl_{model}"][m_metric]
                increment = pl_margin_dct[f"{feature}_{model}"][f"incremental_{m_metric}"]
                error = data[f"{feature}_{model}"][sd_metric]

                # Colors and saturation
                alpha = color_dct["bar_saturation"][model]
                other_feat_color = color_dct["other"]
                pl_color = color_dct["pl"]

                if increment > 0:
                    ax.barh(y_positions[j] + (i * bar_width), base_value, height=bar_width, color=pl_color,
                            align='center', edgecolor=None, alpha=alpha)
                    ax.barh(y_positions[j] + (i * bar_width), increment, left=base_value, height=bar_width,
                            xerr=error, color=other_feat_color, align='center', edgecolor=None, alpha=alpha,
                            capsize=5)
                else:
                    print(f"{feature}_{model}, no increment, only plot blue bar for the combination")
                    real_value = base_value + increment
                    ax.barh(y_positions[j] + (i * bar_width), real_value, height=bar_width, color=pl_color,
                            xerr=error, align='center', edgecolor=None, alpha=alpha, capsize=5)

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

    def format_bar_plot(self,
                        ax: Axes,
                        row_idx: int,
                        y_positions: ndarray,
                        bar_width: float,
                        feature_combo_mapping: dict[str, str],
                        order: list[str],
                        fontsizes: dict[str, int],
                        metric: str,
                        rel: float = None) -> None:
        """
        Formats the bar plot by setting y-ticks, x-axis label based on the metric, and an optional reference line.

        Args:
            ax: Matplotlib Axes object to format.
            y_positions: Array of y positions for the bars.
            row_idx: row index of the subplot.
            bar_width: Width of the bars.
            feature_combo_mapping: Dictionary mapping features to their group labels.
            order: List of features in the desired order for the y-axis.
            fontsizes: Dictionary of font sizes for the plot.
            metric: Metric for the x-axis label ("spearman", "pearson", or "r2").
            rel: Optional float, the reference line value to draw.
        """
        # Set y-ticks to the feature group labels in reversed order
        ax.set_yticks(y_positions + bar_width / 2)

        # Map features to their group labels and format the strings
        feature_combos = [feature_combo_mapping[feature] for feature in order]
        feature_combos_str_format = [self.line_break_strings(combo,
                                                             max_char_on_line=14,
                                                             balance=False,
                                                             split_strng=";",
                                                             force_split_strng=True)  # 14, False
                                     for combo in feature_combos]

        # Display the formatted feature combinations with the sample_size n in italics
        feature_combos_str_format = [
            re.sub(r'\b[nN]\b', r'$\g<0>$', s) if re.search(r'[nN] =', s) else s
            for s in feature_combos_str_format
        ]
        ax.set_yticklabels(feature_combos_str_format,
                           fontsize=fontsizes["tick_params"],
                           horizontalalignment="left")

        # Set and format axes
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: self.format_metric(x)))
        ax.tick_params(axis='x', labelsize=fontsizes["tick_params"])
        ax.tick_params(axis='y', which='major', pad=90, length=0)  # 90, if fontsizes = 15

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
    def format_metric(metric: float) -> Union[str, int]:
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

        rounded_metric = round(metric, 2)
        formatted_metric = f"{rounded_metric:.2f}".lstrip('0').replace('-.', '-0.')

        return formatted_metric

    def apply_formatted_x_axis(self, ax: Axes) -> None:
        """
        Applies the APA-style formatting to the x-axis of a plot.

        Args:
            ax: The Matplotlib Axes object whose x-axis needs formatting.
        """
        formatter = FuncFormatter(lambda x, _: self.format_metric(x))
        ax.xaxis.set_major_formatter(formatter)

    def create_grid(self,
                    num_rows: int,
                    num_cols: int,
                    figsize: tuple[int] = (15, 20),
                    empty_cells: Collection[tuple[int]] = None)\
            -> tuple[Figure, ndarray[Axes]]:
        """
        This function creates a flexible grid of subplots with customizable empty cells.

        Args:
            num_row: Number of rows in the grid.
            num_cols: Number of columns in the grid.
            figsize: Figure size for the entire grid (default is (15, 20)).
            empty_cells: List of (row, col) tuples where cells should be empty.

        Returns:
            fig: Matplotlib figure object for the grid.
            axes: Array of axes objects for individual plots.
        """
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        if empty_cells:
            for (row, col) in empty_cells:
                fig.delaxes(axes[row, col])  # Remove the axis for empty cells
        return fig, axes

    def get_margins(
            self,
            cv_results_dct: dict[str, dict[str, float]],
            ref_dct: dict[str, dict[str, float]],
            crit: str,
            samples_to_include: str,
            metric: str,
            ref_feature_combo: str = "pl") -> dict[str, dict[str, float]]:
        """
        Computes the incremental performance difference for all combinations involving 'pl' (e.g., 'pl_srmc')
        compared to the base 'pl' metric for each model. It
            - extracts the base metric and the combined metrics to compute the increment
            - stores the increment and the standard deviation of the combined metric in a dict

        Args:
            cv_results_dct: Dictionary containing metrics for each model and feature combination.
            ref_dct: Dict containing the reference values for both models (e.g., pl_elasticnet).
            crit: e.g., wb_state
            samples_to_include: e.g., all
            metric: The metric to compute the incremental performance difference for.
            ref_feature_combo: The reference feature combination to compare the incremental performance to (default is 'pl').

        Returns:
            dict: Dictionary containing incremental performance differences for each 'pl' combination and model.
        """
        margin_dict = {}
        m_metric = f"m_{metric}"
        sd_metric = f"sd_{metric}"
        ref = f"{ref_feature_combo}_"

        filtered_metrics = self.filter_cv_results_data(
            cv_results_dct=cv_results_dct,
            crit=crit,
            samples_to_include=samples_to_include,
        )

        # Loop over each key in the filtered_metrics that contains ref but is not ref (e.g., pl_srmc vs. pl)
        for key, value in filtered_metrics.items():
            if key.startswith(ref) and key not in ref_dct or key.startswith("all_in"):
                model = key.split('_')[-1]
                base_key = f"{ref}{model}"

                if base_key in ref_dct:
                    incremental_difference = value[m_metric] - ref_dct[base_key][m_metric]
                    margin_dict[key] = {
                        f'incremental_m_{metric}': incremental_difference,
                        sd_metric: value[sd_metric]  # Note: We need the SD of the combined metric for the plots
                    }

        return margin_dict

    @staticmethod
    def get_refs(cv_results_dct: dict[str, dict[str, float]],  # TODO check
                 crit: str,
                 samples_to_include: str,
                 ref_feature_combo: str = "pl") -> dict[str, dict[str, float]]:
        """
        This function extracts the reference values for the two- and three-level analysis in the cv_result plots.

        Args:
            cv_results_dct: Dict containing the M and SD for a given feature_combo/crit/samples_to_include/model combination
                and a predefined metric
            crit: e.g., "wb_state"
            samples_to_include: e.g., "all"
            ref_feature_combo: e.g., "pl"

        Returns:
            dict: Contaning the reference values used in the plot for both models

        """
        return {
            f"{ref_feature_combo}_elasticnet":
                cv_results_dct[f"{crit}_{ref_feature_combo}_elasticnet_{samples_to_include}"],
            f"{ref_feature_combo}_randomforestregressor":
                cv_results_dct[f"{crit}_{ref_feature_combo}_randomforestregressor_{samples_to_include}"]
        }

    def plot_shap_beeswarm_plots(self, prepare_shap_data_func: Callable, prepare_shap_ia_data_func: Callable = None):
        """
        Plots SHAP beeswarm plots for different predictor combinations arranged in a grid.

        Args:
            prepare_shap_data_func:
            prepare_shap_ia_data_func:

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
        beeswarm_title_params = self.plot_cfg["shap_beeswarm_plot"]["titles"]
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
                            feature_combination_to_plot=ia_cfg["feature_combination"],
                            meta_stat_to_extract=ia_cfg["meta_stat_to_extract"],
                            stat_to_extract=ia_cfg["stat_to_extract"],
                            order_to_extract=ia_cfg["order_to_extract"],
                            num_to_extract=ia_cfg["num_to_extract"]
                        )
                        ia_position = tuple(ia_cfg["position"])
                        positions[ia_position] = next(iter(ia_data_current))

                    # Create a grid of subplots with specified empty cells
                    beeswarm_width = beeswarm_figure_params["width"]
                    beeswarm_height = beeswarm_figure_params["height"]
                    fig, axes = self.create_grid(
                        num_rows=beeswarm_figure_params["num_rows"],
                        num_cols=beeswarm_figure_params["num_cols"],
                        figsize=(beeswarm_width, beeswarm_height)
                    )

                    # Iterate over the positions and plot the SHAP beeswarm plots
                    for (row, col), predictor_combination in positions.items():
                        if predictor_combination in data_current:
                            shap_values = data_current[predictor_combination]
                            self.plot_shap_beeswarm(
                                shap_values=shap_values,
                                ax=axes[row, col],
                                row=row,
                                beeswarm_fontsizes=beeswarm_fontsizes,
                                beeswarm_figure_params=beeswarm_figure_params,
                                num_to_display=num_to_display,
                                cmap=cmap,
                                feature_combo_name_mapping=feature_combo_name_mapping,
                                predictor_combination=predictor_combination,
                                title_line_dct=beeswarm_title_params["line_dct"]
                            )
                            # Add the main title (first-row axes only)
                            if row == 0:
                                first_row_heading = beeswarm_title_params["shap_values"][col]
                                axes[row, col].text(
                                    0.32, 1.35,  # 0.35
                                    first_row_heading,
                                    fontsize=beeswarm_fontsizes["main_title"],
                                    fontweight="bold",
                                    ha="center",
                                    va="bottom",
                                    transform=axes[row, col].transAxes,  # Use axes-relative coordinates
                                )
                        else:
                            if ia_data_current:
                                if predictor_combination in ia_data_current:
                                    shap_ia_data = ia_data_current[predictor_combination]
                                    self.plot_shap_beeswarm(
                                        shap_values=shap_ia_data,
                                        ax=axes[row, col],
                                        row=row,
                                        beeswarm_fontsizes=beeswarm_fontsizes,
                                        beeswarm_figure_params=beeswarm_figure_params,
                                        num_to_display=num_to_display,
                                        cmap=cmap,
                                        feature_combo_name_mapping=feature_combo_name_mapping,
                                        predictor_combination=predictor_combination,
                                        ia_values=True,
                                        title_line_dct=beeswarm_title_params["line_dct"]
                                    )
                                    ia_heading = self.plot_cfg["shap_beeswarm_plot"]["titles"]["shap_ia_values"][0]
                                    axes[row, col].text(
                                        0.48, 1.31,
                                        ia_heading,
                                        fontsize=beeswarm_fontsizes["main_title"],
                                        fontweight="bold",
                                        ha="center",  # Align horizontally
                                        va="bottom",  # Align vertically
                                        transform=axes[row, col].transAxes,  # Use axes-relative coordinates
                                    )
                            else:
                                print(f"Predictor combination '{predictor_combination}' not found in shap data or shap ia data")

                    # Hide empty subplot
                    fig.delaxes(axes[2, 2])
                    if not self.plot_cfg["shap_beeswarm_plot"]["shap_ia_values"]["add"]:
                        fig.delaxes(axes[3, 2])

                    # Adjust layout and display the figure
                    plt.subplots_adjust(
                        top=beeswarm_subplot_adj["top"],
                        left=beeswarm_subplot_adj["left"],
                        wspace=beeswarm_subplot_adj["wspace"],
                        hspace=beeswarm_subplot_adj["hspace"],
                        right=beeswarm_subplot_adj["right"]
                    )

                    if self.store_plots:
                        self.store_plot(
                            plot_name=f"beeswarm_{crit}_{samples_to_include}_{model}",
                            crit=crit,
                            samples_to_include=samples_to_include,
                            plot_format="pdf",  # "pdf"
                            dpi=450,
                            model=model
                        )
                    else:
                        plt.show()

    def plot_shap_beeswarm(self,
                           shap_values: shap.Explanation,
                           ax,
                           row,
                           beeswarm_fontsizes: dict,
                           beeswarm_figure_params: dict,
                           num_to_display: int,
                           cmap,
                           feature_combo_name_mapping,
                           predictor_combination,
                           title_line_dct: dict = None,
                           ia_values: bool = False,
                           ):
        """

        Returns:

        """
        plt.sca(ax)
        if ia_values:
            # max_char_on_line = 44
            # split_string = "x"

            # Currently, we just use the abbreviations # TODO make more flexible
            feature_names_formatted = [f"IA{num_ia}" for num_ia in range(1, num_to_display + 1)]

            # Step 1: Compute the mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            # Step 2: Get the indices that would sort the features by mean absolute SHAP values in descending order
            sorted_indices = np.argsort(mean_abs_shap)[::-1][:num_to_display]

            # Step 3: Select the top features and corresponding SHAP values and data
            shap_values_to_plot = shap_values.values[:, sorted_indices]
            data_to_plot = shap_values.data[:, sorted_indices]

        else:
            max_char_on_line = beeswarm_figure_params["max_char_on_line_y_ticks"]
            split_string = None
            feature_names_formatted = [self.line_break_strings(strng=feature_name,
                                                               max_char_on_line=max_char_on_line,
                                                               balance=True,
                                                               split_strng=split_string)
                                       for feature_name in shap_values.feature_names]
            shap_values_to_plot = shap_values.values
            data_to_plot = shap_values.data

        # Plot the SHAP beeswarm plot in the specified subplot
        shap.summary_plot(
            shap_values_to_plot,
            data_to_plot,
            feature_names=feature_names_formatted,
            max_display=num_to_display,
            show=False,
            plot_size=None,
            color_bar=False,
            cmap=cmap,
        )
        # Set title and fontsizes
        if ia_values:
            split_strng = self.plot_cfg["shap_beeswarm_plot"]["shap_ia_values"]["title_split_strng"]  # : TODO: Check if it works
        else:
            split_strng = self.plot_cfg["shap_beeswarm_plot"]["titles"]["split_strng"]  # +

        formatted_title = self.line_break_strings(
            strng=feature_combo_name_mapping[predictor_combination],
            max_char_on_line=self.plot_cfg["shap_beeswarm_plot"]["titles"]["max_char_on_line"],
            split_strng=split_strng,
        )

        if self.plot_cfg["shap_beeswarm_plot"]["titles"]["add_n"]:
            n_sample_mapping = self.plot_cfg["shap_beeswarm_plot"]["n_samples_mapping"]
            n_samples_formatted = n_sample_mapping[predictor_combination]

            print(n_samples_formatted)
            n_samples_formatted = re.sub(r'\bn\b', r'$n$', n_samples_formatted, flags=re.IGNORECASE)
            formatted_title += f"\n{n_samples_formatted}"

        # Determine the maximum line count for the current row
        max_lines_in_row = title_line_dct[f"row_{row}"]
        current_title_line_count = formatted_title.count("\n") + 1
        y_position = 1.0 + 0.05 * (max_lines_in_row - current_title_line_count)

        ax.set_title(
            formatted_title,
            fontsize=beeswarm_fontsizes["title"],
            y=y_position
        )

        # Align fontsizes and labels
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=beeswarm_fontsizes["tick_params"]
        )
        if ia_values:
            ax.set_xlabel("SHAP IA value (impact on model output)", fontsize=beeswarm_fontsizes["x_label"])  # Custom x-axis label
        ax.xaxis.label.set_size(beeswarm_fontsizes["x_label"])
        ax.yaxis.label.set_size(beeswarm_fontsizes["y_label"])

        # Set xlim to a fix value
        if self.plot_cfg["shap_beeswarm_plot"]["figure"]["fix_x_lim"]:
            ax.set_xlim(tuple(self.plot_cfg["shap_beeswarm_plot"]["figure"]["x_lim"]))

    def get_n_most_important_features(self, shap_values: Explanation, n: int) -> tuple[ndarray, ndarray, list[str]]:
        """
        This function extracts the n most important features from the SHAP values. It
            - gets the abbreviation for plotting SHAP values from the config
            - computes the mean absolute SHAP values / SHAP IA values for each feature
            - gets the indices for the n most important features
            - select the top features and corresponding SHAP values, data, and feature names

        Args:
            shap_values: Explanation objects for a given analysis setting
            n: number of features to extract

        Returns:
            tuple Containing the shape values to plot, the data to plot, and the formatted feature names
        """
        abbr = self.plot_cfg["shap_beeswarm_plot"]["shap_ia_values"]["abbr"]
        feature_names_formatted = [f"{abbr}{num_ia}" for num_ia in range(1, n + 1)]  # TODO, other plots?

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1][:n]

        shap_values_to_plot = shap_values.values[:, sorted_indices]
        data_to_plot = shap_values.data[:, sorted_indices]

        return shap_values_to_plot, data_to_plot, feature_names_formatted

    def plot_pred_true_parity(self,
                              sample_data: dict[str, dict[str, float]],
                              samples_to_include: str,
                              crit: str,
                              model: str) -> None:
        """
        This function creates a parity plot of predicted vs. true values for all samples. This can
        be used to analyze unexpected patterns in the prediction results.

        Args:
            sample_data (dict): Nested dictionary that contain predicted and true values for each sample
            samples_to_include: e.g., "all"
            crit: e.g., "wb_state"
            model: e.g., "randomforestregressor"
        """
        # Create figure
        width = self.plot_cfg["pred_true_parity_plot"]["figure"]["width"]
        height = self.plot_cfg["pred_true_parity_plot"]["figure"]["height"]
        fig, ax = plt.subplots(figsize=(width, height))

        # Assign colors to samples
        samples = list(sample_data.keys())
        num_samples = len(samples)
        colors = self.plot_cfg["custom_cmap_colors"]
        colors = [colors[i % len(colors)] for i in range(num_samples)]

        # Plot scatter plot
        for i, sample_name in enumerate(samples):
            color = colors[i % 10]
            pred_values = sample_data[sample_name]['pred']
            true_values = sample_data[sample_name]['true']
            ax.scatter(true_values, pred_values, color=color, label=sample_name)

        # Plot y = x line for reference
        min_val = min([min(sample_data[s]['true'] + sample_data[s]['pred']) for s in samples])
        max_val = max([max(sample_data[s]['true'] + sample_data[s]['pred']) for s in samples])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')

        # Set label, axes, title
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'Pred vs True - {samples_to_include} - {crit} - {model}')
        ax.legend()
        if self.store_plots:
            self.store_plot(plot_name="pred_vs_true_scatter", crit=crit, samples_to_include=samples_to_include, model=model)
        else:
            plt.show()

    def line_break_strings(
            self,
            strng: str,
            max_char_on_line: int,
            split_strng: str = None,
            balance: bool = False,
            force_split_strng: bool = False
    ) -> str:
        """
        This function recursively breaks a string into multiple lines based on a maximum character limit.
        This is used in the plots to create a balanced layout. Multiple parameters can be adjusted for customization.

        Args:
            strng: The input string to be broken into lines.
            max_char_on_line: Maximum number of characters allowed per line.
            balance (bool, optional): If True, attempts to balance lines for better readability.
            split_strng: If provided, the function aims to split at this string if possible.
            force_split_strng: If True, enforces splitting at `split_strng` for the last line regardless of line length.

        Returns:
            str: The formatted string with line breaks.
        """
        if len(strng) <= max_char_on_line:
            if force_split_strng and split_strng and split_strng in strng:
                # Attempt to split at the last occurrence of split_strng
                split_pos = strng.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                    first_line = strng[:break_pos].rstrip()
                    remainder = strng[break_pos:].lstrip()
                    if remainder:
                        return first_line + '\n' + remainder
                    else:
                        return first_line
            return strng  # Return as-is if no forced split is needed

        # Consider a substring up to max_char_on_line + 1 to include potential split_strng
        substring = strng[:max_char_on_line + 1]

        # Determine where to split
        if split_strng:
            if force_split_strng:
                # Always attempt to split at split_strng
                split_pos = substring.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                else:
                    # If split_strng not found in substring, fallback to space or max_char_on_line
                    split_pos = substring.rfind(' ')
                    break_pos = split_pos if split_pos != -1 else max_char_on_line
            else:
                # If not force_split_strng: prefer split_strng, then space, then max_char_on_line
                split_pos = substring.rfind(split_strng)
                if split_pos != -1:
                    break_pos = split_pos + len(split_strng)
                else:
                    split_pos = substring.rfind(' ')
                    break_pos = split_pos if split_pos != -1 else max_char_on_line
        else:
            # If no split_strng provided, split at the last space or max_char_on_line
            split_pos = substring.rfind(' ')
            break_pos = split_pos if split_pos != -1 else max_char_on_line

        # Balancing Logic
        if balance:
            first_line = strng[:break_pos].rstrip()
            remainder = strng[break_pos:].lstrip()

            if remainder:
                current_diff = abs(len(first_line) - len(remainder))

                best_pos = break_pos
                best_diff = current_diff

                search_range = 10
                start_search = max(break_pos - search_range, 1)
                end_search = min(break_pos + search_range, len(strng) - 1)

                for candidate_pos in range(start_search, end_search + 1):
                    if candidate_pos != break_pos and candidate_pos < len(strng):
                        if strng[candidate_pos] == ' ' or (
                                split_strng and
                                strng[candidate_pos - len(split_strng) + 1:candidate_pos + 1] == split_strng
                        ):
                            test_first = strng[:candidate_pos].rstrip()
                            test_rem = strng[candidate_pos:].lstrip()
                            new_diff = abs(len(test_first) - len(test_rem))
                            if new_diff < best_diff:
                                best_diff = new_diff
                                best_pos = candidate_pos

                break_pos = best_pos
                first_line = strng[:break_pos].rstrip()
                remainder = strng[break_pos:].lstrip()
        else:
            first_line = strng[:break_pos].rstrip()
            remainder = strng[break_pos:].lstrip()

        next_force_split_strng = True

        return first_line + '\n' + self.line_break_strings(
            remainder,
            max_char_on_line,
            split_strng,
            balance,
            force_split_strng=next_force_split_strng
        )

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
