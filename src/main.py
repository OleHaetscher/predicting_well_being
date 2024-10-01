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

        if "cocoesm" in datasets_included:
            coco_esm_preprocessor = CocoesmPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_cocoesm = coco_esm_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_cocoesm)
        if "cocoms" in datasets_included:
            coco_ms_preprocessor = CocomsPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_cocoms = coco_ms_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_cocoms)
        if "cocout" in datasets_included:
            coco_ut_preprocessor = CocoutPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_cocout = coco_ut_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_cocout)
        if "emotions" in datasets_included:
            emotions_preprocessor = EmotionsPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_emotions = emotions_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_emotions)
        if "pia" in datasets_included:
            pia_preprocessor = PiaPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_pia = pia_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_pia)
        if "zpid" in datasets_included:
            zpid_preprocessor = ZpidPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
            df_zpid = zpid_preprocessor.apply_preprocessing_methods()
            df_lst.append(df_zpid)
        # TODO: Make finale variable checks when sensing vars are complete
        df = pd.concat(df_lst, axis=0, ignore_index=False, sort=False, join='outer')

        if var_cfg["preprocessing"]["store_data"]:
            with open("../data/preprocessed/full_data", "wb") as f:
                pickle.dump(df, f)

    if var_cfg["general"]["steps"]["analysis"]:
        if var_cfg["analysis"]["load_data"]:
            df = pd.read_pickle(os.path.join(var_cfg["preprocessing"]["path_to_preprocessed_data"], "full_data"))
            print()
        # args = get_slurm_vars(var_cfg)
        # updated_config = update_cfg_with_slurm_vars(cfg=config, args=args)
        var_cfg_updated = var_cfg

        if os.getenv("SLURM_JOB_ID"):
            #updated_config = sanity_checks_cfg_cluster(updated_config)
            total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            print("total_cores in main:", total_cores)
            #updated_config = allocate_cores_and_update_config(
            #    updated_config, total_cores
            #)
            #output_dir = args.output
        else:
            #output_dir = construct_output_path(updated_config)

            #if not os.path.exists(output_dir):
            #    os.makedirs(output_dir)
            output_dir = var_cfg["analysis"]["output_base_path"]

        prediction_model = var_cfg_updated["analysis"]["params"]["prediction_model"]
        if prediction_model == "enr":
            enr_analyzer = ENRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df)
            enr_analyzer.apply_methods()
        elif prediction_model == "rfr":
            rfr_analyzer = RFRAnalyzer(var_cfg=var_cfg_updated, output_dir=output_dir, df=df)
            rfr_analyzer.apply_methods()
        else:
            raise ValueError(f"Model {prediction_model} not implemented")






















