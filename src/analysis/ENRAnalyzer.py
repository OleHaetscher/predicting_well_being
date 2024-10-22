from collections import defaultdict

from sklearn.linear_model import ElasticNet

from src.analysis.BaseMLAnalyzer import BaseMLAnalyzer


class ENRAnalyzer(BaseMLAnalyzer):
    """
    This class serves as a template for the linear models (lasso, linear_baseline_model) and implements methods
    that do not differ between the both linear models. Inherits from BaseMLAnalyzer. For attributes, see
    BaseMLAnalyzer. The model attribute is defined in the subclasses.
    """

    def __init__(self, var_cfg, output_dir, df, rank):
        """
        Constructor method of the LinearAnalyzer class.

        Args:
            var_cfg: YAML config determining specifics of the analysis
            output_dir: Specific directory where the results are stored
        """
        super().__init__(var_cfg, output_dir, df, rank)
        self.model = ElasticNet(random_state=self.var_cfg["analysis"]["random_state"])

    def get_average_coefficients(self):
        """Calculate the average coefficients across all outer cv loops stored in self.best_models."""
        if self.rank == 0:
            feature_names = self.X.columns.drop(self.meta_vars).tolist()
            # feature_names.remove(self.meta_vars)
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
