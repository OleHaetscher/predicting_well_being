import os

import numpy as np
import shapiq
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer


class RFRAnalyzer(BaseMLAnalyzer):
    """
    This class is the specific implementation of the random forest regression using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Inherits from
    BaseMLAnalyzer. For class attributes, see BaseMLAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, var_cfg, output_dir, df, comm, rank):
        """
        Constructor method of the RFRAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(var_cfg, output_dir, df, comm, rank)
        self.model = RandomForestRegressor(
            random_state=self.var_cfg["analysis"]["random_state"]
        )

    def calculate_shap_ia_values(self, X, pipeline):
        # TODO Adjust
        """
        This method computes SHAP interaction values

        Args:
            X:
            pipeline:

        Returns:

        """
        columns = X.columns
        X_processed = pipeline.named_steps["preprocess"].transform(X)  # Still need this for scaling

        # SHAP IA Value computations
        shap_iq_tree_explainer = shapiq.TreeExplainer(
            model=pipeline.named_steps["model"].regressor_,
            index="k-SII",
            min_order=self.var_cfg["analysis"]["shap_ia_values"]["min_order"],
            max_order=self.var_cfg["analysis"]["shap_ia_values"]["max_order"],
            budget=self.var_cfg["analysis"]["shap_ia_values"]["budget"],
        )
        # Parallelize the calculations processing chunks of the data
        n_jobs = self.var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"]
        chunk_size = X_processed.shape[0] // n_jobs + (X_processed.shape[0] % n_jobs > 0)
        print("n_jobs shap ia _values")
        print("chunk_size:", chunk_size)
        results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(
            delayed(self.compute_shap_ia_values_for_chunk)(
                shap_iq_tree_explainer, X_processed[i: i + chunk_size]
            )
            for i in range(0, X_processed.shape[0], chunk_size)
        )
        print("len results:", len(results))

        shap_ia_values_array = np.vstack(results)
        return shap_ia_values_array

    def compute_shap_ia_values_for_chunk(self, explainer, X_subset):
        # TODO Adjust this to SHAP-IQ
        """
        This function computes (pairwise) SHAP interaction values for the rfr

        Args:
            explainer: shap.TreeExplainer
            X_subset: df, subset of X for which interaction values are computed, can be parallelized

        Returns:
            explainer.shap_interaction_values(X_subset): array containing SHAP interaction values for X_subset
        """
        self.logger.log(f"Currently processing these indices: {X_subset.index[:3]}")
        self.logger.log(f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}")
        ia_value_lst = []
        for idx in X_subset.index:
            x = X_subset[idx]
            ia_value_lst.append(explainer.explain(x))
        return ia_value_lst
