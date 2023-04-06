import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather_feat(feat, ind, mask=None):
    # feat [b, h * w, c]
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()  # [b,c,h,w]->[b,h,w,c]
    feat = feat.view(feat.size(0), -1, feat.size(3))  # [b,h,w,c]->[b, h*w, c]
    feat = _gather_feat(feat, ind)  # ind: b*k;  feat: [b, h*w, c]->[b, k, c]
    return feat


def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(0,
                            w * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    shifts_y = torch.arange(0,
                            h * stride,
                            step=stride,
                            dtype=torch.float32,
                            device=device)
    # w 152
    # h 100
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    # 15200
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
