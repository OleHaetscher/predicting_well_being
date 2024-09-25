import yaml

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

    # Apply preprocessing for the individual datasets
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

    df_all = pd.concat(df_lst, axis=0, ignore_index=False, sort=False, join='outer')











