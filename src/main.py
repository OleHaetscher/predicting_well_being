import yaml

from src.preprocessing.CocoesmPreprocessor import CocoesmPreprocessor
from src.preprocessing.CocomsPreprocessor import CocomsPreprocessor
from src.preprocessing.CocoutPreprocessor import CocoutPreprocessor
from src.preprocessing.DatasetCreator import DatasetCreator
from src.preprocessing.EmotionsPreprocessor import EmotionsPreprocessor
from src.preprocessing.PiaPreprocessor import PiaPreprocessor
from src.preprocessing.ZpidPreprocessor import ZpidPreprocessor

if __name__ == "__main__":
    var_config_path = "../configs/config_var.yaml"
    with open(var_config_path, "r") as f:
        var_cfg = yaml.safe_load(f)
    fix_config_path = "../configs/config_fix.yaml"
    with open(fix_config_path, "r") as f:
        fix_cfg = yaml.safe_load(f)

    # Apply preprocessing for the individual datasets
    datasets_included = var_cfg["general"]["datasets_to_be_included"]

    if "cocoesm" in datasets_included:
        coco_esm_preprocessor = CocoesmPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        df_coco_esm = coco_esm_preprocessor.apply_preprocessing_methods()
        print()

    if "cocoms" in datasets_included:
        coco_ms_preprocessor = CocomsPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        coco_ms_preprocessor.apply_preprocessing_methods()

    if "cocout" in datasets_included:
        coco_ut_preprocessor = CocoutPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        coco_ut_preprocessor.apply_preprocessing_methods()

    if "emotions" in datasets_included:
        emotions_preprocessor = EmotionsPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        emotions_preprocessor.apply_preprocessing_methods()

    if "pia" in datasets_included:
        pia_preprocessor = PiaPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        pia_preprocessor.apply_preprocessing_methods()

    if "zpid" in datasets_included:
        zpid_preprocessor = ZpidPreprocessor(fix_cfg=fix_cfg, var_cfg=var_cfg)
        zpid_preprocessor.apply_preprocessing_methods()

    # Create full DataFrame  # TODO: Refactor? Take variable number of positional arguments with unpacking?
    #dataset_creator = DatasetCreator(data_cocoesm=coco_esm_preprocessor.data,
    #                                 data_cocoms=coco_ms_preprocessor.data,
    #                                 data_cocout=coco_ut_preprocessor.data,
    #                                 data_emotions=emotions_preprocessor.data,
    #                                 data_pia=pia_preprocessor.data,
    #                                 data_zpid=zpid_preprocessor.data
    #                                 )









