from src.postprocessing.ShapProcessor import ShapProcessor
from src.postprocessing.SignificanceTesting import SignificanceTesting
from src.postprocessing.DescriptiveStatistics import DescriptiveStatistics
from src.utils.DataLoader import DataLoader


class Postprocessor:
    """
    This class executes the different postprocessing steps. This includes
        - conducting tests of significance to compare prediction results
        - calculate and display descriptive statistics
        - creates plots (results, SHAP, SHAP interaction values)
    """

    def __init__(self, fix_cfg, var_cfg, name_mapping):
        self.fix_cfg = fix_cfg
        self.var_cfg = var_cfg
        self.name_mapping = name_mapping
        self.data_loader = DataLoader()
        self.descriptives_creator = DescriptiveStatistics(fix_cfg=self.fix_cfg, var_cfg=self.var_cfg, name_mapping=name_mapping)
        self.significance_testing = SignificanceTesting(var_cfg=self.var_cfg)
        self.shap_processor = ShapProcessor(var_cfg=self.var_cfg)

    def postprocess(self):
        self.shap_processor.recreate_explanation_objects()  # TODO Just for testing
        self.descriptives_creator.create_m_sd_feature_table()
        self.descriptives_creator.create_wb_item_statistics()
        self.significance_testing.apply_methods()
