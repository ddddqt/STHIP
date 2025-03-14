from torch import nn
import torch
import numpy as np

from hit.modeling.backbone import build_backbone
from hit.modeling.roi_heads.roi_heads_3d import build_3d_roi_heads
# print("4")
# from hit.parsing.ParsingTransform import ParsingTransform
# from hit.modeling.parsing.build_parsing import build_parsing
# from hit.modeling.parsing.build_parsing_fusion import build_parsing_fusion

class ActionDetector(nn.Module):
    def __init__(self, cfg):
        super(ActionDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_3d_roi_heads(cfg, self.backbone.dim_out)
        # self.parsing = build_parsing(cfg)
        self.parsing_use = cfg.DATASETS.PARSING_USE
        # self.parsing_slowfast = build_parsing_fusion(cfg)

    def forward(self, slow_video, fast_video, boxes, objects=None, keypoints=None, extras={}, part_forward=-1):

        if part_forward == 1:
            slow_features = fast_features = None
        else:
            slow_features, fast_features = self.backbone(slow_video, fast_video)

        parsing_features = None
        if part_forward == 1:
            parsing_features = None
        else:
            if self.parsing_use == True:
                # parsing_slow = ParsingTransform(slow_video, aug=self.cfg.DATASETS.PARSING_AUG)
                # parsing_fast = ParsingTransform(fast_video, aug=self.cfg.DATASETS.PARSING_AUG)
                # parsing = torch.stack((parsing_slow, parsing_fast), dim=1)
                # print("parsing_1",parsin)

                if self.cfg.PARSING.STRAGE == "fusion_in_slowfast_out":
                    # parsing_features = self.parsing(parsing)
                    slow_features,fast_features = self.parsing_slowfast(slow_features, fast_features, parsing_features)
                # elif self.cfg.PARSING.STRAGE == "parsingconv_fusion_in_slowfast_out":
                #     parsing_features = self.parsing(parsing)
                #     print("parsing",parsing.shape)
                #     slow_features,fast_features = self.parsing_slowfast(slow_features, fast_features, parsing_features)
                # elif self.cfg.PARSING.STRAGE == "noparsing_fusion_in_slowfast_out":
                #     # parsing = torch.stack((parsing_slow,parsing_fast))
                #     slow_features,fast_features = self.parsing_slowfast(slow_features, fast_features, parsing)

        result, detector_losses, loss_weight, detector_metrics = self.roi_heads(slow_features, fast_features, boxes, objects, keypoints, extras, part_forward, parsing_features)

        if self.training:
            return detector_losses, loss_weight, detector_metrics, result

        return result

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        new_key = name + '.' + key
                        weight_map[new_key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping

def build_detection_model(cfg):
    return ActionDetector(cfg)