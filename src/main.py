import os
import pickle
import yaml
import numpy as np

# Import MPI
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Hello from rank {rank} out of {size}", flush=True)
print(f"Starting main.py on rank {rank} out of {size}")

from src.analysis.ENRAnalyzer import ENRAnalyzer
from src.analysis.RFRAnalyzer import RFRAnalyzer
from src.postprocessing.Postprocessor import Postprocessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.preprocessing.CocoesmPreprocessor import CocoesmPreprocessor
from src.preprocessing.CocomsPreprocessor import CocomsPreprocessor
from src.preprocessing.CocoutPreprocessor import CocoutPreprocessor
from src.preprocessing.EmotionsPreprocessor import EmotionsPreprocessor
from src.preprocessing.PiaPreprocessor import PiaPreprocessor
from src.preprocessing.ZpidPreprocessor import ZpidPreprocessor
import pandas as pd

from src.utils.SlurmHandler import SlurmHandler

if __name__ == "__main__":
    # Load configurations (all ranks need var_cfg, fix_cfg, name_mapping)
    var_config_path = "../configs/config_var.yaml"
    with open(var_config_path, "r") as f:
        var_cfg = yaml.safe_load(f)
    fix_config_path = "../configs/config_fix.yaml"
    with open(fix_config_path, "r") as f:
        fix_cfg = yaml.safe_load(f)
    name_mapping_path = "../configs/name_mapping.yaml"
    with open(name_mapping_path, "r") as f:
        name_mapping = yaml.safe_load(f)

    # Preprocessing step (all ranks execute)
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
        print()

        if var_cfg["preprocessing"]["store_data"]:
            with open("../data/preprocessed/full_data", "wb") as f:
                pickle.dump(df, f)
    else:
        df = None  # Or handle accordingly if no preprocessing is done

    # Analysis step (all ranks participate)
    if var_cfg["general"]["steps"]["analysis"]:
        slurm_handler = SlurmHandler()

        # Load data if required (all ranks need to load the data)
        if var_cfg["analysis"]["load_data"]:
            # Each rank loads the data from the shared filesystem
            df = pd.read_pickle(os.path.join(var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))
        else:
            # Data already loaded during preprocessing
            pass  # df is already available

        # Handle SLURM job variables (all ranks need to process this)
        if os.getenv("SLURM_JOB_ID"):
            args = slurm_handler.get_slurm_vars(var_cfg=var_cfg)
            var_cfg_updated = slurm_handler.update_cfg_with_slurm_vars(var_cfg=var_cfg, args=args)
            var_cfg_updated = slurm_handler.sanity_checks_cfg_cluster(var_cfg=var_cfg_updated)
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            print("total_cores in main:", total_cores)
            var_cfg_updated = slurm_handler.allocate_cores(var_cfg=var_cfg_updated, total_cores=total_cores)
            output_dir = args.output_path
        else:
            # Non-SLURM environment
            var_cfg_updated = var_cfg
            output_dir = slurm_handler.construct_local_output_path(var_cfg=var_cfg)

        # All ranks proceed with analysis
        np.random.seed(var_cfg["analysis"]["random_state"])

        prediction_model = var_cfg_updated["analysis"]["params"]["prediction_model"]
        if prediction_model == "elasticnet":
            enr_analyzer = ENRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df, rank=rank)
            enr_analyzer.apply_methods(comm=comm)
        elif prediction_model == "randomforestregressor":
            rfr_analyzer = RFRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df, rank=rank)
            rfr_analyzer.apply_methods(comm=comm)
        else:
            raise ValueError(f"Model {prediction_model} not implemented")

    # Postprocessing step (only executed on rank 0)
    if var_cfg["general"]["steps"]["postprocessing"]:
        postprocessor = Postprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg, name_mapping=name_mapping)
        postprocessor.postprocess()





















