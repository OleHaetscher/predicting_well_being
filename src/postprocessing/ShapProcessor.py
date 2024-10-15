import os

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.utils.DataLoader import DataLoader


# TODO: This should
class ShapProcessor:
    """
    This class processes the SHAP values. It
        - summarizes the SHAP values obtained across outer folds
        - recreates the SHAP explanation objects for plotting
    """

    def __init__(self, var_cfg):
        self.var_cfg = var_cfg
        self.data_loader = DataLoader()

    def recreate_explanation_objects(self):
        """
        This function recreates the SHAP explanation objects from the
            - SHAP value arrays
            - Base value arrays
            - Features
        Returns:
        """
        # TODO This is more of a test script if we can succefully rebuild the Explanation objects -> Adjust
        res_dir = "../results/ml_analysis/pl/selected/trait_pa/randomforestregressor"
        filename = os.path.join(res_dir, "shap_values.pkl")
        shap_results = self.data_loader.read_pkl(filename)

        # Make example for rep0, average across imputations
        shap_vals = shap_results['shap_values']["rep_0"]
        shap_vals = np.mean(shap_vals, axis=2)
        base_vals = shap_results['base_values']["rep_0"]
        base_vals = np.mean(base_vals, axis=1).flatten()
        shap_data = shap_results['data']["rep_0"]
        shap_data = np.mean(shap_data, axis=2)
        features = shap_results["feature_names"]

        features.remove("other_unique_id")
        shap_df = pd.DataFrame(shap_data, columns=features)
        explanation_test = shap.Explanation(shap_vals, base_vals, data=shap_data, feature_names=features)

        shap.summary_plot(explanation_test.values, shap_df)
        shap.plots.waterfall(explanation_test[10])
        shap.plots.violin(explanation_test)
        plt.show()
        print("####")
