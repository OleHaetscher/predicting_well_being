import os
import pickle
import yaml
import numpy as np
import pandas as pd

from src.utils.SlurmHandler import SlurmHandler

if __name__ == "__main__":
    # Load configurations (before importing mpi4py)
    var_config_path = "../configs/config_var.yaml"
    with open(var_config_path, "r") as f:
        var_cfg = yaml.safe_load(f)
    fix_config_path = "../configs/config_fix.yaml"
    with open(fix_config_path, "r") as f:
        fix_cfg = yaml.safe_load(f)
    name_mapping_path = "../configs/name_mapping.yaml"
    with open(name_mapping_path, "r") as f:
        name_mapping = yaml.safe_load(f)

    slurm_handler = SlurmHandler()

    # Parse arguments before importing mpi4py
    args = slurm_handler.get_slurm_vars(var_cfg=var_cfg)
    var_cfg_updated = slurm_handler.update_cfg_with_slurm_vars(var_cfg=var_cfg, args=args)
    use_mpi = var_cfg_updated["analysis"]["use_mpi4py"]
    split_reps = var_cfg_updated["analysis"]["split_reps"]
    rep = args.rep

    # Now, conditionally import
    if use_mpi:
        from mpi4py import MPI
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print("using mpi")
    else:
        comm = None
        rank = 0
        size = 1
        print("not using mpi")

    print(f"Hello from rank {rank} out of {size}", flush=True)
    print(f"Starting main.py on rank {rank} out of {size}")

    # Rest of your imports
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
            df = pd.read_pickle(os.path.join(var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))
        else:
            pass  # df is already available

        # Handle SLURM job variables
        if os.getenv("SLURM_JOB_ID"):
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            print("total_cores in main:", total_cores)
            var_cfg_updated = slurm_handler.allocate_cores(var_cfg=var_cfg_updated, total_cores=total_cores)
            var_cfg_updated = slurm_handler.sanity_checks_cfg_cluster(var_cfg=var_cfg_updated)
            output_dir = args.output_path
            if not use_mpi and split_reps and args.rep is not None:
                rep = args.rep
            else:
                rep = None
        else:
            output_dir = slurm_handler.construct_local_output_path(var_cfg=var_cfg_updated)
            rep = None

        # All ranks proceed with analysis
        np.random.seed(var_cfg["analysis"]["random_state"])

        prediction_model = var_cfg_updated["analysis"]["params"]["prediction_model"]
        if prediction_model == "elasticnet":
            enr_analyzer = ENRAnalyzer(
                var_cfg=var_cfg_updated,
                output_dir=output_dir,
                df=df,
                rank=rank,
                rep=rep  # Pass the repetition number
            )
            enr_analyzer.apply_methods(comm=comm)
        elif prediction_model == "randomforestregressor":
            rfr_analyzer = RFRAnalyzer(
                var_cfg=var_cfg_updated,
                output_dir=output_dir,
                df=df,
                rank=rank,
                rep=rep  # Pass the repetition number
            )
            rfr_analyzer.apply_methods(comm=comm)
        else:
            raise ValueError(f"Model {prediction_model} not implemented")

    # Postprocessing step
    if var_cfg_updated["general"]["steps"]["postprocessing"]:
        if use_mpi and rank == 0:
            postprocessor = Postprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg_updated, name_mapping=name_mapping)
            postprocessor.postprocess()
        elif not use_mpi:
            # Run postprocessing after all repetitions have been completed
            postprocessor = Postprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg_updated, name_mapping=name_mapping)
            postprocessor.postprocess()





















