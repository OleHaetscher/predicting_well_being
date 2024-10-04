import json
import os
from collections import defaultdict

import numpy as np
import shap
from joblib import Parallel, delayed

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer
from sklearn.linear_model import ElasticNet

from src.analysis.Imputer import Imputer


class ENRAnalyzer(BaseMLAnalyzer):
    """
    This class serves as a template for the linear models (lasso, linear_baseline_model) and implements methods
    that do not differ between the both linear models. Inherits from BaseMLAnalyzer. For attributes, see
    BaseMLAnalyzer. The model attribute is defined in the subclasses.
    """

    def __init__(self, var_cfg, output_dir, df):
        """
        Constructor method of the LinearAnalyzer class.

        Args:
            var_cfg: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(var_cfg, output_dir, df)
        self.model = ElasticNet(random_state=self.var_cfg["analysis"]["random_state"])

    def get_average_coefficients(self):
        """Calculate the average coefficients across all outer cv loops stored in self.best_models."""
        feature_names = self.X.columns.tolist()
        feature_names.remove(self.id_grouping_col)
        coefs_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for rep in range(self.num_reps):
            for outer_fold_idx, outer_fold in enumerate(self.best_models[f"rep_{rep}"]):
                for imputation_idx, model in enumerate(outer_fold):
                    print(rep, outer_fold_idx, imputation_idx)
                    coefs_sub_dict = dict(zip(feature_names, model.coef_))
                    sorted_coefs_sub_dict = dict(
                        sorted(
                            coefs_sub_dict.items(), key=lambda item: abs(item[1]), reverse=True
                        )
                    )
                    # Insert sorted_coefs_dict into coefs_dict according to the hierarchy
                    coefs_dict[f"rep_{rep}"][f"outer_fold_{outer_fold_idx}"][f"imputation_{imputation_idx}"] = sorted_coefs_sub_dict

        regular_dict = self.defaultdict_to_dict(coefs_dict)
        self.lin_model_coefs = regular_dict

    def defaultdict_to_dict(self, dct):
        if isinstance(dct, defaultdict):
            dct = {k: self.defaultdict_to_dict(v) for k, v in dct.items()}
        return dct

    def calculate_shap_for_instance(self, n_instance, instance, explainer):
        """
        Calculates linear SHAP values for a single instance for parallelization.

        Args:
            n_instance: Number of a certain individual to calculate SHAP values for
            instance: 1d-array, represents the feature values for a single individual
            explainer: shap.LinearExplainer

        Returns:
            explainer(instance.reshape(1, -1)).values: array containing the SHAP values for "n_instance"
        """
        if n_instance < 3:
            self.log_thread()
        return (explainer(instance.reshape(1, -1)).values,
                explainer(instance.reshape(1, -1)).base_values,
                explainer(instance.reshape(1, -1)).data)

    def calculate_shap_values(self, X, pipeline):
        """
        This function calculates linear SHAP values for a given analysis setting. This includes applying the
        preprocessing steps that were applied in the pipeline (e.g., scaling, RFECV if specified).
        It calculates the SHAP values using the explainers.Linear. SHAP calculations can be parallelized.

        Args:
            X: df, features for the machine learning analysis according to the current specification
            pipeline: Sklearn Pipeline object containing the steps of the ml-based prediction (i.e., preprocessing
                and estimation using the prediction model).

        Returns:
            shap_values_array: ndarray, obtained SHAP values, of shape (n_features x n_samples)
            columns: pd.Index, contains the names of the features in X associated with the SHAP values
            shap_interaction_values: None, returned here for method consistency
        """
        columns = X.columns
        X_processed = pipeline.named_steps["preprocess"].transform(X)  # must contain imputations

        explainer_lin = shap.LinearExplainer(
            pipeline.named_steps["model"].regressor_, X_processed
        )

        results = Parallel(
            n_jobs=self.var_cfg["analysis"]["parallelize"]["shap_n_jobs"], verbose=0
        )(
            delayed(self.calculate_shap_for_instance)(
                n_instance, instance, explainer_lin
            )
            for n_instance, instance in enumerate(X_processed)
        )
        shap_values_lst, base_values_lst, data_lst = zip(*results)

        # Convert list of arrays to single array
        shap_values_array = np.vstack(shap_values_lst)
        base_values_array = np.array(base_values_lst)
        data_array = np.vstack(data_lst)
        shap_interaction_values = None
        return shap_values_array, base_values_array, data_array, columns, shap_interaction_values
