import os
import pdb
from collections import OrderedDict

import torch

replace_dict = OrderedDict(
    {
        "controller_attention.feature_embed0": "feature_interaction.feature_embed",
        "proposal_generator.classification_head": "feature_interaction.cas_module.block",
        "controller_attention": "feature_interaction",
    }
)
delete_keys = [
    "matcher.controller_attention.decoder.layers.0.q_proj.weight",
    "matcher.controller_attention.decoder.layers.0.k_proj.weight",
    "matcher.controller_attention.decoder.layers.0.v_proj.weight",
    "matcher.controller_attention.decoder.layers.0.merge.weight",
    "matcher.controller_attention.decoder.layers.1.q_proj.weight",
    "matcher.controller_attention.decoder.layers.1.k_proj.weight",
    "matcher.controller_attention.decoder.layers.1.v_proj.weight",
    "matcher.controller_attention.decoder.layers.1.merge.weight",
]


def key_in_replace_dict(key, r_dict):
    for rk in r_dict.keys():
        if rk in key:
            return True, rk, r_dict[rk]
    return False, None, None


origin_ckpt_path = "./datasets/ft_local/m2o_0225_dump/best_model/epoch=29-auc@5=0.511-auc@10=0.686-auc@20=0.813.ckpt"
target_ckpt_path = "weights/adamatcher.ckpt"
origin_ckpt = torch.load(origin_ckpt_path, map_location="cpu")
origin_state_dict = origin_ckpt["state_dict"]
target_state_dict = OrderedDict()
for key in origin_state_dict.keys():
    if key in delete_keys:
        continue
    replace_flag, replace_key, new_key = key_in_replace_dict(key, replace_dict)
    if replace_flag:
        target_state_dict[key.replace(replace_key, new_key)] = origin_state_dict[key]
    else:
        target_state_dict[key] = origin_state_dict[key]

targect_ckpt = origin_ckpt.copy()
targect_ckpt["state_dict"] = target_state_dict
torch.save(targect_ckpt, target_ckpt_path)
