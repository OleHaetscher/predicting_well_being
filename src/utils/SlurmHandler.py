import argparse
import os


class SlurmHandler:
    """
    Class that implements methods to handle the computation on the cluster using SLURM scheduler

    """

    def get_slurm_vars(self, var_cfg):
        """
        This function updates the args that were provided by the SLURM script.
        Args:
            var_cfg: Dict, containing the yaml var_cfg for default arguments for the parameters
        Returns:
            args: argparse.Namespace, contains the SLURM arguments passed to the script
        """
        parser = argparse.ArgumentParser(description="CoCo WB ML - OH")

        # Add string arguments
        parser.add_argument("--prediction_model", type=str, help="elasticnet or randomforestregressor")
        parser.add_argument("--crit", type=str, help="state or trait wb/pa/na")
        parser.add_argument("--feature_combination", type=str, help="Feature combinations defined in the PreReg")
        parser.add_argument("--samples_to_include", type=str, help="all, selected, control")
        parser.add_argument("--output_path", type=str, help="Output file path.")

        # Add boolean arguments using str2bool
        parser.add_argument("--comp_shap_ia_values", type=self.str2bool, help="Calculate IA values for RFR, Bool")
        parser.add_argument("--parallelize_imputation_runs", type=self.str2bool, help="Parallelize imputed datasets, Bool")
        parser.add_argument("--parallelize_inner_cv", type=self.str2bool, help="Parallelize inner CV, Bool")
        parser.add_argument("--parallelize_shap_ia_values", type=self.str2bool, help="Parallelize SHAP IA values, Bool")
        parser.add_argument("--parallelize_shap", type=self.str2bool, help="Parallelize SHAP calculations, Bool")
        parser.add_argument("--use_mpi", type=self.str2bool, help="Use mpi4py, Bool")
        parser.add_argument("--split_reps", type=self.str2bool, help="Split repetitions into separate jobs, Bool")

        # Add integer argument for repetitions
        parser.add_argument("--rep", type=int, default=None, help="Repetition number, used when split_reps is true")

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
        This function updates the current var_cfg with the SLURM vars provided.
        Args:
            var_cfg: Dict, containing the old YAML var_cfg before parameter update
            args: argparse.Namespace, contains the SLURM arguments passed to the script
        Returns:
            cfg_updated: Dict, containing the new YAML var_cfg with the parameters defined in the SLURM script
        """
        args_dict = vars(args)
        print(args_dict)
        for arg_name, arg_value in args_dict.items():
            # Skip 'rep' as it's not part of var_cfg
            if arg_name == "rep" or arg_value is None:
                continue
            updated = False
            # Update 'general' section
            if arg_name in ["use_mpi", "split_reps"]:
                var_cfg["analysis"][arg_name] = arg_value
                updated = True
            # Update 'params' section
            elif arg_name in var_cfg["analysis"]["params"]:
                var_cfg["analysis"]["params"][arg_name] = arg_value
                updated = True
            # Update 'parallelize' section
            elif arg_name in var_cfg["analysis"]["parallelize"]:
                var_cfg["analysis"]["parallelize"][arg_name] = arg_value
                updated = True
            # Update 'shap_ia_values' section
            elif arg_name in var_cfg["analysis"]["shap_ia_values"]:
                var_cfg["analysis"]["shap_ia_values"][arg_name] = arg_value
                updated = True
            # Update 'output_path'
            elif arg_name == "output_path":
                var_cfg["analysis"]["output_path"] = arg_value
                updated = True
            else:
                print(f"Warning: Argument {arg_name} not recognized. Skipping update for this argument.")

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
        else:
            var_cfg["analysis"]["parallelize"]["inner_cv_n_jobs"] = 1

        if var_cfg["analysis"]["parallelize"]["parallelize_imputation_runs"]:
            var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"] = total_cores
        else:
            var_cfg["analysis"]["parallelize"]["imputation_runs_n_jobs"] = 1


        if var_cfg["analysis"]["parallelize"]["parallelize_shap"]:
            var_cfg["analysis"]["parallelize"]["shap_n_jobs"] = total_cores
        else:
            var_cfg["analysis"]["parallelize"]["shap_n_jobs"] = 1

        if var_cfg["analysis"]["parallelize"]["parallelize_shap_ia_values"]:
            var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"] = total_cores
        else:
            var_cfg["analysis"]["parallelize"]["shap_ia_values_n_jobs"] = 1

        return var_cfg

    @staticmethod
    def sanity_checks_cfg_cluster(var_cfg):
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
        if "store_analysis_results" not in var_cfg["analysis"]["methods_to_apply"]:
            var_cfg["analysis"]["methods_to_apply"].append("store_analysis_results")

        # for safety, adjust number of reps and outer cvs
        var_cfg["analysis"]["cv"]["num_inner_cv"] = 10
        var_cfg["analysis"]["cv"]["num_outer_cv"] = 10
        var_cfg["analysis"]["cv"]["num_reps"] = 10
        var_cfg["analysis"]["imputation"]["num_imputations"] = 5
        var_cfg["analysis"]["imputation"]["max_iter"] = 40

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
