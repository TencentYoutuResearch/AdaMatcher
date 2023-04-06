import pdb

import torch
import torch.nn as nn
from einops.einops import rearrange
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
from torchsummaryX import summary

from src.adamatcher.backbone import build_proposal_generator
from src.adamatcher.backbone.feature_interaction import FICAS

h = 480 // 8
w = 640 // 8


class FICAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.controller_attention = FICAS()
        self.proposal_generator = build_proposal_generator()

    def forward(self, x0, x1, m0=None, m1=None):
        # x:[n, (h*w), c]  m:[n, h, w]
        x0, x1, hs0, hs1, x0_tmp, x1_tmp = self.controller_attention(
            x0, x1, m0, m1)
        x0_tmp = rearrange(x0_tmp, 'n (h w) c -> n c h w', h=h,
                           w=w).contiguous()
        x1_tmp = rearrange(x1_tmp, 'n (h w) c -> n c h w', h=h,
                           w=w).contiguous()

        # classification0, attn_mask0 = self.proposal_generator(x0_tmp, hs0, m0,
        #                                                       h=480, w=640)
        # classification1, attn_mask1 = self.proposal_generator(x1_tmp, hs1, m1,
        #                                                       h=480, w=640)
        classification, attn_mask = self.proposal_generator(
            torch.cat([x0_tmp, x1_tmp], dim=0),
            torch.cat([hs0, hs1], dim=0),
            None,  # torch.cat([m0, m1], dim=0),
            h=h,
            w=w,
        )
        classification0, classification1 = classification.split(1)
        return classification0, classification1, x0, x1


# pdb.set_trace()
# device = torch.device('cuda:1')
device = torch.device('cpu')
model = FICAS().to(device).eval()
# x0 = torch.randn(1, 480*640, 256)
# x1 = torch.randn(1, 480*640, 256)
# m0 = torch.randn(1, 480, 640)
# m1 = torch.randn(1, 480, 640)

# flops, params = profile(model, inputs=(x0, x1, m0, m1))
# print('FLOPs = ' + str(2*flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

# summary(model, torch.randn(1, 480*640, 256), torch.randn(1, 480*640, 256),
#         torch.randn(1, 480, 640), torch.randn(1, 480, 640))

# input = (torch.rand(1, 480*640, 256), torch.rand(1, 480*640, 256),
#         torch.ones(1, 480, 640).bool(), torch.rand(1, 480, 640).bool())
input = (
    torch.rand(1, h * w, 256).to(device),
    torch.rand(1, h * w, 256).to(device),
)
flops = FlopCountAnalysis(model, input)
# print(parameter_count_table(model))
print('FLOPs: ', flops.total() / 1e9)


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


print(print_model_parm_nums(model))
