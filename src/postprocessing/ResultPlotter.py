import matplotlib.pyplot as plt
import numpy as np
import shap


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
    def __init__(self, var_cfg):
        self.var_cfg = var_cfg

    def plot_cv_results(self, processed_cv_result_dict: dict) -> None:
        """
        This function creates plots of the results.
            - For each criterion (e.g., 'state_wb')
            - For each condition (e.g., 'control')
            - In one big plot, it plots all different feature combinations (e.g., 'pl_srmc_mac')
            - In this big plot, it plots the mean and sd across outer folds and imputations for the 'r2' metric
            - For both 'elasticnet' and 'random forest', in subplots

        Args:
            processed_cv_result_dict (dict): Nested dict containing the cv_results for all analysis
        """
        # TODO This will be refined later
        # Collect all conditions and criteria from the result_stats
        conditions = set()
        criteria = set()
        models = set()
        for fc_dict in processed_cv_result_dict.values():
            for condition, cond_dict in fc_dict.items():
                conditions.add(condition)
                for criterion, crit_dict in cond_dict.items():
                    criteria.add(criterion)
                    models.update(crit_dict.keys())

        models = list(models)

        # Iterate over each condition and criterion
        for condition in conditions:
            for criterion in criteria:
                # Initialize data containers for plotting
                feature_combinations = []
                data_per_model = {model: {'means': [], 'sds': []} for model in models}

                # Collect data for each feature combination
                for feature_combination, fc_dict in processed_cv_result_dict.items():
                    if condition in fc_dict and criterion in fc_dict[condition]:
                        crit_dict = fc_dict[condition][criterion]
                        for model in models:
                            if model in crit_dict:
                                stats = crit_dict[model]
                                mean_r2 = stats['m']['r2']
                                sd_r2 = stats['sd_across_folds_imps']['r2']
                                data_per_model[model]['means'].append(mean_r2)
                                data_per_model[model]['sds'].append(sd_r2)
                        feature_combinations.append(feature_combination)

                if not feature_combinations:
                    continue  # Skip if no data is available for this condition and criterion

                num_models = len(models)
                fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), sharey=True)
                if num_models == 1:
                    axes = [axes]

                x = np.arange(len(feature_combinations))
                for i, model in enumerate(models):
                    ax = axes[i]
                    means = data_per_model[model]['means']
                    sds = data_per_model[model]['sds']
                    if means:
                        ax.bar(x, means, yerr=sds, align='center', alpha=0.7, ecolor='black', capsize=5)
                        ax.set_xticks(x)
                        ax.set_xticklabels(feature_combinations, rotation=45, ha='right')
                        ax.set_ylabel('R2 Score')
                        ax.set_title(f'{model}')
                        ax.yaxis.grid(True)
                    else:
                        ax.set_visible(False)

                fig.suptitle(f'Criterion: {criterion} - Condition: {condition}', fontsize=16)
                # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
                plt.show()

    def create_grid(self, num_rows: int, num_cols: int, figsize=(25, 20), empty_cells=None):
        """
        This function creates a flexible grid of subplots.

        Args:
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
            figsize (tuple): Figure size for the entire grid (default is (15, 10)).
            empty_cells (list of tuples): List of (row, col) tuples where cells should be empty.

        Returns:
            fig (plt.Figure): Matplotlib figure object for the grid.
            axes (np.ndarray): Array of axes objects for individual plots.
        """
        # Initialize figure and axes grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

        # Flatten axes array for easier access to individual cells
        axes = axes.flatten()

        # Set specified cells as empty
        if empty_cells:
            for (row, col) in empty_cells:
                cell_idx = row * num_cols + col
                fig.delaxes(axes[cell_idx])  # Remove the axis for empty cells

        # Reshape axes back to original shape if needed for consistent indexing
        axes = axes.reshape(num_rows, num_cols)

        # Return figure and axes array
        return fig, axes

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
        fig, axes = self.create_grid(num_rows=4, num_cols=3, figsize=(20,15), empty_cells=[(2, 2), (3, 2)])

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

                ### TODO Remove and find fault
                feature_names =shap_values.feature_names.copy()
                if "Power distance" in feature_names:
                    reordered_feature_names = feature_names
                    #index = feature_names.index("Power distance")
                    #reordered_feature_names = feature_names[index:] + feature_names[:index]
                    #print("changed feature order for", predictor_combination)
                else:
                    reordered_feature_names = feature_names

                shap_values = shap.Explanation(values=shap_values.values,
                                               base_values=shap_values.base_values,
                                               data=shap_values.data,
                                               feature_names=reordered_feature_names)

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
        plot_name = f"{crit}_{model}_imp.png"
        plt.savefig(plot_name, format='png', dpi=300)  # You can adjust the format and DPI as needed
        plt.close()

    def plot_shap_beeswarm_plot(self, data: dict, crit: str, samples_to_include: str, model: str):
        """
        Plots SHAP beeswarm plots for different predictor combinations arranged in a grid.

        Args:
            data: Nested dictionary containing shap_values.
            crit: Criterion to filter data.
            samples_to_include: Which samples to include.
            model: Model name.

        Returns:
            None
        """
        # Extract the relevant data
        data_current = data[crit][samples_to_include][model]

        # Create a grid of subplots with specified empty cells
        fig, axes = self.create_grid(num_rows=4, num_cols=3, figsize=(20, 30), empty_cells=[(2, 2), (3, 2)])

        # fig.set_size_inches(18, 24)
        # Define the arrangement of predictor combinations in the grid
        first_col = ["pl", "srmc", "sens", "mac"]
        second_col = ["pl_srmc", "pl_sens", "pl_srmc_sens", "pl_mac"]
        third_col = ["pl_srmc_mac", "all_in"]

        # Define the predictor combinations
        predictor_combinations = [
            "pl", "srmc", "sens", "mac",
            "pl_srmc", "pl_sens", "pl_srmc_sens", "pl_mac",
            "pl_srmc_mac", "all_in"
        ]

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

        # Iterate over the positions and plot the SHAP beeswarm plots
        for (row, col), predictor_combination in positions.items():
            if predictor_combination in data_current:
                shap_values = data_current[predictor_combination]
                ax = axes[row][col]
                # Set the current axis to the subplot
                plt.sca(ax)
                # Plot the SHAP beeswarm plot in the specified subplot
                shap.plots.beeswarm(
                    shap_values,
                    max_display=5,
                    show=False,
                    color_bar=False,
                    plot_size=None,
                    alpha=0.6,
                    s=3
                )
                # Set the title of the subplot to the predictor combination name
                ax.set_title(predictor_combination, fontsize=22)
                ax.tick_params(axis='both', which='major', labelsize=13)
            else:
                print(f"Predictor combination '{predictor_combination}' not found in data.")

        # Hide any unused subplots
        for idx in range(len(predictor_combinations), len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout and display the figure
        plt.subplots_adjust(left=0.25, wspace=0.3, hspace=0.8)  # Adjust horizontal and vertical spacing
        plot_name = f"{crit}_{model}_bee.png"
        plt.savefig(plot_name, format='png', dpi=300)  # You can adjust the format and DPI as needed
        plt.close()

    def plot_lin_model_coefs(self):
        pass

    def plot_shap_values(self):
        pass

    def plot_shap_ia_values(self):
        pass
