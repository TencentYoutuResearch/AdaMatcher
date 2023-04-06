import argparse
import os
import pdb

# os.chdir("..")
from copy import deepcopy

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from src.adamatcher import AdaMatcher
from src.adamatcher.localization.satr import Matcher
from src.config.default import get_cfg_defaults
from src.datasets.megadepth import read_megadepth_color

parser = argparse.ArgumentParser(description="Localize Aachen Day-Night")
parser.add_argument("--gpu", "-gpu", type=str, default=2)
# parser.add_argument('--colmap', type=str, default='colmap')
parser.add_argument(
    "--ckpt",
    type=str,
    default="./weights/adamatcher.ckpt",
)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

main_cfg_path = "configs/loftr/outdoor/loftr_ds_dense.py"
data_root = "./datasets/megadepth/test"
img0_pth = "Undistorted_SfM/0015/images_1349576135_91cfa7d80a_o.jpg"
img1_pth = "Undistorted_SfM/0015/images_2362907762_bb9469a630_o.jpg"
out_dir = "./heatmap"
os.makedirs(out_dir, exist_ok=True)

# config = get_cfg_defaults()
# config.merge_from_file(main_cfg_path)
matcher = Matcher(args)

image0, mask0, scale0, scale_wh0 = read_megadepth_color(
    img0_pth, resize=832, df=64, padding=True, augment_fn=None
)
mask0_d8 = F.interpolate(
    image0[None, None].float(),
    scale_factor=1 / matcher.scale_l1,
    mode="nearest",
    recompute_scale_factor=False,
)[0].bool()
mask0_d64 = F.interpolate(
    image0[None, None].float(),
    scale_factor=1 / matcher.scale_l0,
    mode="nearest",
    recompute_scale_factor=False,
)[0].bool()
image0, mask0 = image0.to(matcher.device), mask0.to(matcher.device)

image1, mask1, scale1, scale_wh1 = read_megadepth_color(
    img1_pth, resize=832, df=64, padding=True, augment_fn=None
)
mask1_d8 = F.interpolate(
    image1[None, None].float(),
    scale_factor=1 / matcher.scale_l1,
    mode="nearest",
    recompute_scale_factor=False,
)[0].bool()
mask1_d64 = F.interpolate(
    image1[None, None].float(),
    scale_factor=1 / matcher.scale_l0,
    mode="nearest",
    recompute_scale_factor=False,
)[0].bool()
image1, mask1 = image1.to(matcher.device), mask1.to(matcher.device)

with torch.no_grad():
    (
        feat0_d8,
        feat0_d2,
        pred_class0_l1,
        pred_class0_l0,
    ) = matcher.extract_feature_and_mask(image0[None], mask0_d8, mask0_d64)
    (
        feat1_d8,
        feat1_d2,
        pred_class1_l1,
        pred_class1_l0,
    ) = matcher.extract_feature_and_mask(image1[None], mask1_d8, mask1_d64)

    n, c, h0, w0 = feat0_d8.shape
    feat0 = rearrange(matcher.model.pos_encoding(feat0_d8), "n c h w -> n (h w) c")
    feat1 = rearrange(matcher.model.pos_encoding(feat1_d8), "n c h w -> n (h w) c")
    mask0_d8, mask1_d8 = mask1_d8.flatten(-1), mask0_d8.flatten(-1)
    for i, (layer, name) in enumerate(
        zip(
            matcher.model.feature_interaction.layers1,
            matcher.model.feature_interaction.layer_names1,
        )
    ):
        x0, x1 = matcher.model.feature_interaction.transformer(
            feat0, feat1, mask0_d8, mask1_d8, name, layer
        )

    # stage 2
    feature_embed0 = matcher.model.feature_interaction.feature_embed0.weight.unsqueeze(
        0
    ).repeat(
        1, 1, 1
    )  # [bs, num_q, c]
    feature_embed1 = matcher.model.feature_interaction.feature_embed1.weight.unsqueeze(
        0
    ).repeat(
        1, 1, 1
    )  # [bs, num_q, c]
    tgt0 = torch.zeros_like(feature_embed0)
    tgt1 = torch.zeros_like(feature_embed1)
    hs0 = matcher.model.feature_interaction.decoder(
        tgt0, x0, tgt_mask=None, memory_mask=mask0_d8, tgt_pos=feature_embed0
    )
    hs1 = matcher.model.feature_interaction.decoder(
        tgt1, x1, tgt_mask=None, memory_mask=mask1_d8, tgt_pos=feature_embed1
    )
    pdb.set_trace()
    heatmap0_0 = torch.einsum("nlc,nkc->nlk", x0, hs0).squeeze(0).softmax(dim=0)
    heatmap0_0 = rearrange(heatmap0_0, "(h w) c -> h w c", h=h0, w=w0).cpu().numpy()

    fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    axes.imshow(heatmap0_0)
    pair_name = (
        img0_pth.replace("/", "_").split(".")[0]
        + "-"
        + img1_pth.replace("/", "_").split(".")[0]
    )
    path = os.path.join(out_dir, pair_name + ".jpg")
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()

    for i, (layer, name) in enumerate(
        zip(
            matcher.model.feature_interaction.layers2,
            matcher.model.feature_interaction.layer_names2,
        )
    ):
        x0, hs1 = matcher.model.feature_interaction.transformer(
            x0, hs1, mask0_d8, None, name, layer
        )
        x1, hs0 = matcher.model.feature_interaction.transformer(
            x1, hs0, mask1_d8, None, name, layer
        )
    heatmap0_1 = torch.einsum("nlc,nkc->nlk", x0, hs0)

    # stage 3
    for i, (layer, name) in enumerate(
        zip(
            matcher.model.feature_interaction.layers3,
            matcher.model.feature_interaction.layer_names3,
        )
    ):
        x0, x1 = matcher.model.feature_interaction.transformer(
            x0, x1, mask0_d8, mask1_d8, name, layer
        )
    heatmap0_2 = torch.einsum("nlc,nkc->nlk", x0, hs0)
