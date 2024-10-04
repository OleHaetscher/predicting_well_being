import argparse
import os


class SlurmHandler:
    """
    Class that implements methods to handle the computation on the cluster using SLURM scheduler

    """

    def get_slurm_vars(self, var_cfg):
        """
        This function is used to update the args that were provided by the SLURM script. Thus, in the SLURM scipt,
        we provide arguments that determine the current analysis that is run on the cluster (i.e., a specific analysis /
        study / esm sample/ etc) combination. This parameters has to be passed to the python script that runs the
        machine learning analysis. This is done via this function using an ArgumentParser object.

        Args:
            var_cfg: Dict, containing the yaml var_cfg for default arguments for the parameters

        Returns:
            args: argparse.Namespace, contains the SLURM arguments passed to the script
        """
        # Dictionary of arguments
        args_dict = {
            "--prediction_model": {
                "help": "elasticnet or randomforestregression",
            },
            "--crit": {
                "help": "state or trait wb/pa/na",
            },
            "--feature_combination": {
                "help": "all combinations of pl, srmc, sens, and mac defined in the PreReg",
            },
            "--samples_to_include": {
                "help": "all (include all samples and impute a lot), "
                        "selected (include selected samples),"
                        "control (include only pl for the samples included in selected",
            },
            "--output_path": {"default": "test_results", "help": "output file path."},
        }

        parser = argparse.ArgumentParser(
            description="CoCo WB ML - OH"
        )

        # Loop through the dictionary and add each argument
        for arg, params in args_dict.items():
            parser.add_argument(
                arg, type=str, help=params["help"]
            )

        # In the get_slurm_vars function, when adding arguments, specify the type as str2bool for boolean variables
        parser.add_argument(
            "--comp_shap_ia_values",
            type=self.str2bool,
            help="if for rfr ia_values are calculated, Bool",
        )
        parser.add_argument(
            "--parallelize_imputations",
            type=self.str2bool,
            help="if we parallelize the creation of the imputed datasets, Bool",
        )
        parser.add_argument(
            "--parallelize_inner_cv",
            type=self.str2bool,
            help="if we parallelize the inner cv of the analysis, Bool",
        )
        parser.add_argument(
            "--parallelize_shap_ia_values",
            type=self.str2bool,
            help="if we parallelize the shap ia value calculations, Bool",
        )
        parser.add_argument(
            "--parallelize_shap",
            type=self.str2bool,
            help="if we parallelize the shap calculations, Bool",
        )
        args = parser.parse_args()
        return args

    @staticmethod
    def str2bool(v):
        """
        Convert a string representation of truth to true (1) or false (0).
        Accepts 'yes', 'true', 't', 'y', '1' as true and 'no', 'false', 'f', 'n', '0' as false.
        Raises ValueError if 'v' is anything else.

        Args:
            v: [bool, str, num], a certain value provied that should be converted to the boolean equivalent

        Returns:
            [v, True, False]: bool, the boolean expression for the given input
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise ValueError("Boolean value expected.")

    @staticmethod
    def update_cfg_with_slurm_vars(var_cfg, args):
        """
        This function updates the currenct var_cfg with the SLURM vars provided so that the machine learning analysis
        can still grab the parameters from the var_cfg, but with the updated parameters of the current analysis.
        It is important though that this updated var_cfg is used in the main script, not the old var_cfg.

        Args:
            var_cfg: Dict, containg the old YAML var_cfg before parameter update
            args:

        Returns:
            cfg_updated: Dict, cpontaining the new YAML var_cfg with the parameters defined in the SLURM script
        """
        args_dict = vars(args)
        for arg_name, arg_value in args_dict.items():
            print(f"Argument {arg_name}: {arg_value}")
            if arg_name in var_cfg["analysis"]["params"]:
                var_cfg["analysis"]["params"][arg_name] = arg_value
            elif arg_name in var_cfg["analysis"]["parallelize"]:
                var_cfg["analysis"]["parallelize"][arg_name] = arg_value
            elif arg_name == "comp_shap_ia_values":
                var_cfg["analysis"]["shap_ia_values"]["comp_shap_ia_values"] = arg_value
            elif arg_name == "output_path":
                var_cfg["analysis"]["output_path"] = arg_value
            else:
                raise ValueError(f"Argument {arg_name} not recognized.")
        return var_cfg

    @staticmethod
    def allocate_cores(var_cfg, total_cores):
        """
        This function allocates a given number of CPUs on the cluster (as defined in the SLURM script) to the different
        task that are computed during the machine learning analysis. This allows different levels of parallelism.
        I tried this to check what yields the highest computation efficiency. How to parallelize is determined
        by the given var_cfg.

        Args:
            var_cfg: Dict, containing the YAML var_cfg
            total_cores: int, number of cores avaiable in the current analysis on the supercomputer cluster

        Returns:
            var_cfg: Dict, containg the YAML var_cfg where the cores per tasks are included
        """
        print("total_cores in func:", total_cores)
        if var_cfg["analysis"]["parallelize"]["parallelize_inner_cv"]:
            var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"] = total_cores
        if var_cfg["analysis"]["parallelize"]["parallelize_imputations"]:
            var_cfg["analysis"]["parallelize"]["imputations_n_jobs"] = total_cores
        if var_cfg["analysis"]["parallelize"]["parallelize_shap"]:
            var_cfg["analysis"]["parallelize"]["shap_n_jobs"] = total_cores
        if var_cfg["analysis"]["parallelize"]["parallelize_shap_ia_values"]:
            var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"] = total_cores
        return var_cfg

    @staticmethod
    def sanity_checks_cfg_cluster(var_cfg):  # TODO This does not work yet
        """
        This function sets certain variables automatically when using the cluster which I probably change
        locally during testing. For example, for testing certain analysis settings, I might run the CV
        procedure with num_cv=3, because this does not take too much time locally. If I forget to re-adjust
        this parameter in the var_cfg again before runnin the analysis on the cluster, this is done automatically
        in this function.

        Args:
            var_cfg: Dict, containing the YAML var_cfg

        Returns:
            var_cfg: Dict, containing the updated YAML var_cfg
        """
        if "store_analysis_results" not in var_cfg["analysis"]["machine_learning_methods"]:
            var_cfg["analysis"]["machine_learning_methods"].append("store_analysis_results")

        # for safety, adjust number of reps and outer cvs
        #var_cfg["analysis"]["cv"]["num_inner_cv"] = 10
        #var_cfg["analysis"]["cv"]["num_outer_cv"] = 10
        #var_cfg["analysis"]["cv"]["num_reps"] = 10
        #var_cfg["analysis"]["imputation"]["num_imputations"] = 5

        # for safety, adjust the method -> only machine learning is done on the cluster
        var_cfg["general"]["steps"]["preprocessing"] = False
        var_cfg["general"]["analysis"] = True
        var_cfg["general"]["steps"]["postprocessing"] = False
        return var_cfg

    @staticmethod
    def construct_local_output_path(var_cfg):
        """
        This function constructs the path were the results for the current analysis are stored. I used this only
        when I ran analyses locally, otherwise the SLURM script creates the result directory.

        Args:
            var_cfg: Dict, containing the YAML config

        Returns:
            local_output_path: str, the Path were the results for the current ML analysis are stored
        """
        base_output_dir = var_cfg["analysis"]["output_path"]
        crit = var_cfg["analysis"]["params"]["crit"]
        feature_combination = var_cfg["analysis"]["params"]["feature_combination"]
        samples_to_include = var_cfg["analysis"]["params"]["samples_to_include"]
        prediction_model = var_cfg["analysis"]["params"]["prediction_model"]

        local_output_path = os.path.normpath(os.path.join(
            base_output_dir,
            feature_combination,
            samples_to_include,
            crit,
            prediction_model)
        )
        if not os.path.exists(local_output_path):
            os.makedirs(local_output_path, exist_ok=True)
        return local_output_path
