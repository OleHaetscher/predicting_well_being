import os

import numpy as np
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer

from src.analysis.Imputer import Imputer


class RFRAnalyzer(BaseMLAnalyzer):
    """
    This class is the specific implementation of the random forest regression using the standard Sklearn implementation
    (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html). Inherits from
    BaseMLAnalyzer. For class attributes, see BaseMLAnalyzer. Hyperparameters to tune are defined in the config.
    """

    def __init__(self, var_cfg, output_dir, df):
        """
        Constructor method of the RFRAnalyzer class.

        Args:
            config: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(var_cfg, output_dir, df)
        self.model = RandomForestRegressor(
            random_state=self.var_cfg["analysis"]["random_state"]
        )

    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """Calculates tree-based SHAP values for a single instance for parallelization.

        Args:
            n_instance: Number of a certain individual to calculate SHAP values for
            instance: 1d-array, represents the feature values for a single individual
            explainer: shap.TreeExplainer

        Returns:
            explainer(instance.reshape(1, -1)).values: array containing the SHAP values for "n_instance"
        """
        return (explainer(instance.reshape(1, -1)).values,
                explainer(instance.reshape(1, -1)).base_values,
                explainer(instance.reshape(1, -1)).data)

    def compute_shap_interaction_values(self, explainer, X_subset):
        # TODO Adjust this to SHAP-IQ
        """
        This function computes (pairwise) SHAP interaction values for the rfr

        Args:
            explainer: shap.TreeExplainer
            X_subset: df, subset of X for which interaction values are computed, can be parallelized

        Returns:
            explainer.shap_interaction_values(X_subset): array containing SHAP interaction values for X_subset
        """
        print(
            "Currently processing these indices of the original df:",
            X_subset.index[:10],
        )
        print(
            f"Calculating SHAP for subset of length {len(X_subset)} in process {os.getpid()}"
        )
        return explainer.shap_interaction_values(X_subset)

    def calculate_shap_values(self, X, pipeline):
        """
        This function calculates tree-based SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.TreeExplainer, the SHAP implementation that is
        suitable for tree-based models. SHAP calculations can be parallelized.
        Further, it calculates the SHAP interaction values based on the TreeExplainer, if specified

        Args:
            X: df, features for the machine learning analysis according to the current specification
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: SHAP interaction values, of shape (n_features x n_features x n_samples)
        """
        columns = X.columns
        X_processed = pipeline.named_steps["preprocess"].transform(X)  # Still need this for scaling

        explainer_tree = shap.explainers.Tree(pipeline.named_steps["model"])

        if self.var_cfg["analysis"]["comp_shap_ia_values"]:
            # Parallelize the calculations processing chunks of the data
            n_jobs = self.var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"]
            chunk_size = X.shape[0] // n_jobs + (X.shape[0] % n_jobs > 0)
            print("n_jobs shap ia _values")
            print("chunk_size:", chunk_size)
            results = Parallel(n_jobs=n_jobs, verbose=1, backend="multiprocessing")(
                delayed(self.compute_shap_interaction_values)(
                    explainer_tree, X[i: i + chunk_size]
                )
                for i in range(0, X.shape[0], chunk_size)
            )
            print("len results:", len(results))

            shap_ia_values_array = np.vstack(results)
        else:
            shap_ia_values_array = None

        print(self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"])
        results = Parallel(
            n_jobs=self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"], verbose=0
        )(
            delayed(self.calculate_shap_for_instance)(
                n_instance, instance, explainer_tree
            )
            for n_instance, instance in enumerate(X_processed)
        )
        shap_values_lst, base_values_lst, data_lst = zip(*results)

        # Convert list of arrays to single array
        shap_values_array = np.vstack(shap_values_lst)
        base_values_array = np.array(base_values_lst)
        data_array = np.vstack(data_lst)
        return shap_values_array, base_values_array, data_array, columns, shap_ia_values_array
