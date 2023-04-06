import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from yacs.config import CfgNode as CN

from src.adamatcher.ada_module.transformer import (
    DecoderLayer,
    EncoderLayer,
    TransformerDecoder,
)


def make_head_layer(cnv_dim, curr_dim, out_dim, head_name=None):

    fc = nn.Sequential(
        nn.Conv2d(cnv_dim, curr_dim, kernel_size=3, padding=1, bias=True),
        # nn.BatchNorm2d(curr_dim, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
        nn.Conv2d(curr_dim, out_dim, kernel_size=3, stride=1, padding=1),
    )  # kernel=1, padding=0, bias=True

    for l in fc.modules():
        if isinstance(l, nn.Conv2d):
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    return fc


class FeatureAttention(nn.Module):
    config = CN()
    config.nhead = 8
    config.layer_names = ["self", "cross"] * 4
    config.attention = "linear"

    def __init__(self, layer_num=2, d_model=256):
        super(FeatureAttention, self).__init__()
        self.config.layer_names = ["self", "cross"] * layer_num
        self.d_model = d_model
        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            nhead=self.config.nhead,
            attention=self.config.attention,
        )
        self.layer_names = self.config.layer_names
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x0, x1, x0_mask=None, x1_mask=None, flag=False):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]      ->   # [N, L, C]
            source (torch.Tensor): [N, C, H1, W1] ->   # [N, S, C]
            x_mask (torch.Tensor): [N, H0, W0]       -> # [N, L] (optional)
            source_mask (torch.Tensor): [N, H1, W1]  -> # [N, S] (optional)
        """
        assert self.d_model == x0.size(
            2
        ), "the feature number of src and transformer must be equal"

        if x0_mask != None and x1_mask != None:
            x0_mask, x1_mask = x0_mask.flatten(-2), x1_mask.flatten(-2)

        save_feat = []
        if flag is False:
            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == "self":
                    src0, src1 = x0, x1
                    src0_mask, src1_mask = x0_mask, x1_mask
                elif name == "cross":
                    src0, src1 = x1, x0
                    src0_mask, src1_mask = x1_mask, x0_mask
                else:
                    raise KeyError
                x0 = layer(x0, src0, x0_mask, src0_mask)
                x1 = layer(x1, src1, x1_mask, src1_mask)
                if i == 1:  # i==len(self.layer_names)//2-1:
                    # print(i, len(self.layer_names))
                    save_feat.append((x0, x1))
        elif flag == 1:  # origin
            for layer, name in zip(self.layers, self.layer_names):
                if name == "self":
                    x0 = layer(x0, x0, x0_mask, x0_mask)
                    x1 = layer(x1, x1, x1_mask, x1_mask)
                elif name == "cross":
                    x0 = layer(x0, x1, x0_mask, x1_mask)
                    x1 = layer(x1, x0, x1_mask, x0_mask)
                else:
                    raise KeyError
        elif flag == 2:
            for layer, name in zip(self.layers, self.layer_names):
                if name == "self":
                    x0 = layer(x0, x0, x0_mask, x0_mask)
                    x1 = layer(x1, x1, x1_mask, x1_mask)
                elif name == "cross":
                    x1 = layer(x1, x0, x1_mask, x0_mask)
                    x0 = layer(x0, x1, x0_mask, x1_mask)
                else:
                    raise KeyError

        # return feat0, feat1
        if len(save_feat) > 0:
            return x0, x1, save_feat
        else:
            return x0, x1


class SegmentationModule(nn.Module):
    def __init__(self, d_model, num_query):
        super(SegmentationModule, self).__init__()
        self.num_query = num_query
        self.block = make_head_layer(
            d_model, d_model // 2, 1, head_name="classification"
        )
        # self.bn = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        # self.gamma = nn.Parameter(data=torch.ones(1, ), requires_grad=True)

    def forward(self, x, hs, mask=None):
        # x:[n, 256, h, w]  hs:[n, num_q, 256]

        # TODO: BN
        if mask is not None:
            # hs = self.encoderlayer(hs, x3_flatten, None, mask_flatten)
            attn_mask = torch.einsum("mqc,mchw->mqhw", hs, x)
            # attn_mask = self.bn(attn_mask)
            # attn_mask = attn_mask * self.gamma
            attn_mask = attn_mask.sigmoid() * mask.unsqueeze(1)
            classification = self.block(x * attn_mask + x).sigmoid().squeeze(1) * mask
        else:
            # hs = self.encoderlayer(hs, x3_flatten)
            attn_mask = torch.einsum("mqc,mchw->mqhw", hs, x)
            # attn_mask = self.bn(attn_mask)
            # attn_mask = attn_mask * self.gamma
            attn_mask = attn_mask.sigmoid()
            classification = self.block(x * attn_mask + x).sigmoid().squeeze(1)
        return classification  # , attn_mask # , mask_feat


class FICAS(nn.Module):
    config = CN()
    config.nhead = 8
    config.attention = "linear"

    def __init__(self, layer_num=4, d_model=256):
        super(FICAS, self).__init__()
        self.d_model = d_model
        self.num_query = 1
        self.cas_module = SegmentationModule(d_model, self.num_query)

        encoder_layer = EncoderLayer(
            d_model=self.d_model,
            nhead=self.config.nhead,
            attention=self.config.attention,
        )
        self.layer_names1 = [
            "self",
            "cross",
        ]  # ['self', 'cross', 'cross']  # ['self', 'cross']  origin for eccv
        self.layers1 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names1))]
        )

        self.feature_embed = nn.Embedding(self.num_query, self.d_model)
        decoder_layer = DecoderLayer(
            d_model,
            8,
            dropout=0.1,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=2)
        self.layer_names2 = ["cross"]
        self.layers2 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names2))]
        )

        self.layer_names3 = [
            "self",
            "cross",
        ]  # ['self', 'cross', 'cross']  # ['self', 'cross']  origin for eccv
        self.layers3 = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names3))]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def transformer(self, x0, x1, x0_mask, x1_mask, layer_name, layer):
        if layer_name == "self":
            src0, src1 = x0, x1
            src0_mask, src1_mask = x0_mask, x1_mask
        elif layer_name == "cross":
            src0, src1 = x1, x0
            src0_mask, src1_mask = x1_mask, x0_mask
        else:
            raise KeyError
        if (
            x0.shape == x1.shape
            and src0.shape == src1.shape
            and x0_mask is not None
            and x1_mask is not None
            and src0_mask is not None
            and src1_mask is not None
            and not self.training
            and 0
        ):  #  \
            # and layer_name == 'self' and 0:
            temp_x = layer(
                torch.cat([x0, x1], dim=0),
                torch.cat([src0, src1], dim=0),
                torch.cat([x0_mask, x1_mask], dim=0),
                torch.cat([src0_mask, src1_mask], dim=0),
            )
            x0, x1 = temp_x.split(x0.shape[0])
        else:
            x0 = layer(x0, src0, x0_mask, src0_mask)
            x1 = layer(x1, src1, x1_mask, src1_mask)
        return x0, x1

    def feature_interaction(self, x0, x1, x0_mask=None, x1_mask=None):
        """x (torch.Tensor): [N, L, C] source (torch.Tensor): [N, S, C] x_mask
        (torch.Tensor): [N, H0, W0]       -> # [N, L] (optional) source_mask
        (torch.Tensor): [N, H1, W1]  -> # [N, S] (optional)"""
        bs = x0.size(0)
        assert self.d_model == x0.size(
            2
        ), "the feature number of src and transformer must be equal"
        if x0_mask != None and x1_mask != None:
            x0_mask, x1_mask = x0_mask.flatten(-2), x1_mask.flatten(-2)

        # stage 1
        for i, (layer, name) in enumerate(zip(self.layers1, self.layer_names1)):
            x0, x1 = self.transformer(x0, x1, x0_mask, x1_mask, name, layer)

        # stage 2
        feature_embed0 = self.feature_embed.weight.unsqueeze(0).repeat(
            bs, 1, 1
        )  # [bs, num_q, c]
        feature_embed1 = self.feature_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        tgt0 = torch.zeros_like(feature_embed0)
        tgt1 = torch.zeros_like(feature_embed1)
        # hs0 = self.decoder(tgt0, x0, tgt_mask=None, memory_mask=x0_mask)
        # hs1 = self.decoder(tgt1, x1, tgt_mask=None, memory_mask=x1_mask)
        if (
            0
        ):  # x0.shape==x1.shape and x0_mask is not None and x0_mask.shape==x1_mask.shape:
            hs_o = self.decoder(
                torch.cat([tgt0, tgt1], dim=0),
                torch.cat([x0, x1], dim=0),
                tgt_mask=None,
                memory_mask=torch.cat([x0_mask, x1_mask], dim=0),
                tgt_pos=torch.cat([feature_embed0, feature_embed1], dim=0),
            )
            hs0, hs1 = hs_o.split(bs)
        else:
            hs0 = self.decoder(
                tgt0, x0, tgt_mask=None, memory_mask=x0_mask, tgt_pos=feature_embed0
            )
            hs1 = self.decoder(
                tgt1, x1, tgt_mask=None, memory_mask=x1_mask, tgt_pos=feature_embed1
            )

        for i, (layer, name) in enumerate(zip(self.layers2, self.layer_names2)):
            if not self.training and x0.shape == x1.shape and x0_mask is not None:
                x_, hs_ = self.transformer(
                    torch.cat([x0, x1], dim=0),
                    torch.cat([hs1, hs0], dim=0),
                    torch.cat([x0_mask, x1_mask], dim=0),
                    None,
                    name,
                    layer,
                )
                x0, x1 = x_.split(bs)
                hs1, hs0 = hs_.split(bs)
            else:
                x0, hs1 = self.transformer(x0, hs1, x0_mask, None, name, layer)
                x1, hs0 = self.transformer(x1, hs0, x1_mask, None, name, layer)

        x0_mid = x0
        x1_mid = x1
        # stage 3
        for i, (layer, name) in enumerate(zip(self.layers3, self.layer_names3)):
            x0, x1 = self.transformer(x0, x1, x0_mask, x1_mask, name, layer)

        return x0, x1, hs0, hs1, x0_mid, x1_mid

    def covisible_segment(self, x0_mid, x1_mid, hs0, hs1, x0_mask, x1_mask):
        bs = x0_mid.size(0)
        if (
            x0_mask is not None
            and x1_mask is not None
            and x0_mask.shape == x1_mask.shape
            and not self.training
        ):
            cas_scores = self.cas_module(
                torch.cat([x0_mid, x1_mid], dim=0),
                torch.cat([hs0, hs1], dim=0),
                torch.cat([x0_mask, x1_mask], dim=0),
            )
            cas_score0, cas_score1 = cas_scores.split(bs)
        elif x0_mid.shape == x1_mid.shape and x0_mask is None and not self.training:
            cas_scores = self.cas_module(
                torch.cat([x0_mid, x1_mid], dim=0),
                torch.cat([hs0, hs1], dim=0),
            )
            cas_score0, cas_score1 = cas_scores.split(bs)
        else:
            cas_score0 = self.cas_module(x0_mid, hs0, x0_mask)
            cas_score1 = self.cas_module(x1_mid, hs1, x1_mask)
        return cas_score0, cas_score1

    def forward(self, x0, x1, x0_mask=None, x1_mask=None, use_cas=True):
        h0, w0 = x0.shape[2:]
        h1, w1 = x1.shape[2:]
        bs = x0.shape[0]
        x0 = rearrange(x0, "n c h w -> n (h w) c")
        x1 = rearrange(x1, "n c h w -> n (h w) c")
        out0, out1, hs0, hs1, x0_mid, x1_mid = self.feature_interaction(
            x0, x1, x0_mask, x1_mask
        )
        # out0 = rearrange(out0, 'n (h w) c -> n c h w',
        #                  h=h0, w=w0).contiguous()
        # out1 = rearrange(out1, 'n (h w) c -> n c h w',
        #                  h=h1, w=w1).contiguous()
        if use_cas:
            x0_mid = rearrange(x0_mid, "n (h w) c -> n c h w", h=h0, w=w0).contiguous()
            x1_mid = rearrange(x1_mid, "n (h w) c -> n c h w", h=h1, w=w1).contiguous()

            cas_score0, cas_score1 = self.covisible_segment(
                x0_mid, x1_mid, hs0, hs1, x0_mask, x1_mask
            )
        else:
            cas_score0, cas_score1 = torch.ones((bs, h0, w0)).to(x0), torch.ones(
                (bs, h1, w1)
            ).to(x1)

        return out0, out1, cas_score0, cas_score1
