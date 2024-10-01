import os
import pickle

import yaml

from src.analysis.ENRAnalyzer import ENRAnalyzer
from src.analysis.RFRAnalyzer import RFRAnalyzer
from src.preprocessing.CocoesmPreprocessor import CocoesmPreprocessor
from src.preprocessing.CocomsPreprocessor import CocomsPreprocessor
from src.preprocessing.CocoutPreprocessor import CocoutPreprocessor
from src.preprocessing.EmotionsPreprocessor import EmotionsPreprocessor
from src.preprocessing.PiaPreprocessor import PiaPreprocessor
from src.preprocessing.ZpidPreprocessor import ZpidPreprocessor
import pandas as pd

from src.utils.SlurmHandler import SlurmHandler

if __name__ == "__main__":
    var_config_path = "../configs/config_var.yaml"
    with open(var_config_path, "r") as f:
        var_cfg = yaml.safe_load(f)
    fix_config_path = "../configs/config_fix.yaml"
    with open(fix_config_path, "r") as f:
        fix_cfg = yaml.safe_load(f)

    if var_cfg["general"]["steps"]["preprocessing"]:
        datasets_included = var_cfg["general"]["datasets_to_be_included"]
        df_lst = []
        preprocessor_mapping = {
            "cocoesm": CocoesmPreprocessor,
            "cocoms": CocomsPreprocessor,
            "cocout": CocoutPreprocessor,
            "emotions": EmotionsPreprocessor,
            "pia": PiaPreprocessor,
            "zpid": ZpidPreprocessor
        }
        for dataset_name in datasets_included:
            if dataset_name in preprocessor_mapping:
                preprocessor_class = preprocessor_mapping[dataset_name]
                preprocessor = preprocessor_class(fix_cfg=fix_cfg, var_cfg=var_cfg)
                df = preprocessor.apply_preprocessing_methods()
                df_lst.append(df)
            else:
                raise ValueError(f"Warning: No preprocessor defined for dataset '{dataset_name}'")

        df = pd.concat(df_lst, axis=0, ignore_index=False, sort=False, join='outer')

        if var_cfg["preprocessing"]["store_data"]:
            with open("../data/preprocessed/full_data", "wb") as f:
                pickle.dump(df, f)

    if var_cfg["general"]["steps"]["analysis"]:
        slurm_handler = SlurmHandler()
        if var_cfg["analysis"]["load_data"]:
            df = pd.read_pickle(os.path.join(var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))

        if os.getenv("SLURM_JOB_ID"):
            args = slurm_handler.get_slurm_vars(var_cfg=var_cfg)
            var_cfg_updated = slurm_handler.update_cfg_with_slurm_vars(var_cfg=var_cfg, args=args)
            var_cfg_updated = slurm_handler.sanity_checks_cfg_cluster(var_cfg=var_cfg_updated)
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))  # this got me into trouble when I outsourced it
            print("total_cores in main:", total_cores)
            var_cfg_updated = slurm_handler.allocate_cores(var_cfg=var_cfg_updated, total_cores=total_cores)
            output_dir = args.output_path
        else:
            var_cfg_updated = var_cfg
            output_dir = slurm_handler.construct_local_output_path(var_cfg=var_cfg)

        prediction_model = var_cfg_updated["analysis"]["params"]["prediction_model"]
        if prediction_model == "elasticnet":
            enr_analyzer = ENRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df)
            enr_analyzer.apply_methods()
        elif prediction_model == "randomforestregressor":
            rfr_analyzer = RFRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df)
            rfr_analyzer.apply_methods()
        else:
            raise ValueError(f"Model {prediction_model} not implemented")






















