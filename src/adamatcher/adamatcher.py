import pdb

# from .utils.fine_module2 import FineModule2 as FineModule
import time

import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .backbone.feature_interaction import FICAS
from .utils.coarse_module import CoarseModule
from .utils.fine_module import FineModule
from .utils.position_encoding import PositionEncodingSine


class AdaMatcher(nn.Module):
    def __init__(self, config, training=True):
        super().__init__()
        # Misc
        self.config = config
        self.use_cas = config["use_cas"]
        self.scale_l0, self.scale_l1, self.scale_l2 = config["resolution"]
        self.patch_size = self.scale_l0 // self.scale_l1
        self.training = training

        # Modules
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], max_shape=(512, 512)
        )
        self.backbone = build_backbone(config)  # ResNetFPN_64_8_2
        self.feature_interaction = FICAS()  # FeatureAttention()

        self.coarse_module = CoarseModule(config["match_coarse"], config["resolution"])
        self.fine_module = FineModule(config["resolution"])

    def forward(self, data):
        # 1. Local Feature CNN
        # bs = data['image0'].size(0)
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if self.training and 0:
            (feat_d8_0, feat_d2_0), (feat_d8_1, feat_d2_1) = self.backbone(
                data["image0"]
            ), self.backbone(data["image1"])
        else:
            if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
                p3, p1 = self.backbone(
                    torch.cat([data["image0"], data["image1"]], dim=0)
                )
                (feat_d8_0, feat_d8_1), (feat_d2_0, feat_d2_1) = p3.split(
                    data["bs"]
                ), p1.split(data["bs"])
            else:  # handle different input shapes
                (feat_d8_0, feat_d2_0), (feat_d8_1, feat_d2_1) = self.backbone(
                    data["image0"]
                ), self.backbone(data["image1"])

        data.update(
            {
                "hw0_l0": (
                    feat_d8_0.size(2) // self.patch_size,
                    feat_d8_0.size(3) // self.patch_size,
                ),
                "hw1_l0": (
                    feat_d8_1.size(2) // self.patch_size,
                    feat_d8_1.size(3) // self.patch_size,
                ),
                "hw0_d8": feat_d8_0.shape[2:],
                "hw1_d8": feat_d8_1.shape[2:],
                "hw0_d2": feat_d2_0.shape[2:],
                "hw1_d2": feat_d2_1.shape[2:],
            }
        )

        # 2. coarse-level  module
        feat_c0 = self.pos_encoding(feat_d8_0)
        feat_c1 = self.pos_encoding(feat_d8_1)
        mask_feat0, mask_feat1, cas_score0, cas_score1 = self.feature_interaction(
            feat_c0,
            feat_c1,
            data.get("mask0_d8", None),
            data.get("mask1_d8", None),
            use_cas=self.use_cas or self.training,
        )

        feat_c_0, feat_c_1 = mask_feat0, mask_feat1
        mask_feat0 = rearrange(
            mask_feat0, "n (h w) c -> n c h w", h=data["hw0_d8"][0], w=data["hw0_d8"][1]
        ).contiguous()
        mask_feat1 = rearrange(
            mask_feat1, "n (h w) c -> n c h w", h=data["hw1_d8"][0], w=data["hw1_d8"][1]
        ).contiguous()

        data.update(
            {"cas_score0": cas_score0, "cas_score1": cas_score1}  # [N, h0_l1, w0_l1]
        )  # [N, h1_l1, w1_l1]

        # torch.cuda.synchronize()
        # self.ficas_time += time.time() - t1

        # coarse match
        self.coarse_module(data, mask_feat0, mask_feat1)

        # sub-pixel refinement
        # feat_d2_0 = self.pos_encoding_fine(feat_d2_0)
        # feat_d2_1 = self.pos_encoding_fine(feat_d2_1)
        self.fine_module(data, feat_d2_0, feat_d2_1, feat_c_0, feat_c_1)
