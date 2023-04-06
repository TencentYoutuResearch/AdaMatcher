import pdb
from math import log

import torch
from einops import rearrange, repeat
from kornia.utils import create_meshgrid
from loguru import logger

from src.utils.metrics import symmetric_epipolar_distance

from .geometry import pose2essential_fundamental, warp_kpts

##############  ↓  Coarse-Level supervision  ↓  ##############
INF = 100000000


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images."""
    mask = repeat(mask, "n h w -> n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale = config["ADAMATCHER"]["RESOLUTION"][0]
    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale0" in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    )  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    )
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0" in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data["mask0"])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data["mask1"])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i, _ = warp_kpts(
        grid_pt0_i,
        data["depth0"],
        data["depth1"],
        data["T_0to1"],
        None,
        data["K0"],
        data["K1"],
    )
    _, w_pt1_i, _ = warp_kpts(
        grid_pt1_i,
        data["depth1"],
        data["depth0"],
        data["T_1to0"],
        None,
        data["K1"],
        data["K0"],
    )
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (
            (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
        )

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack(
        [nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0
    )
    correct_0to1 = loop_back == torch.arange(h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({"conf_matrix_gt": conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids})

    # 6. save intermediate results (for fast fine-level computation)
    data.update({"spv_w_pt0_i": w_pt0_i, "spv_pt1_i": grid_pt1_i})


@torch.no_grad()
def get_warp_index(
    bs_kpts0, bs_kpts1, E_0to1, E_1to0, K0, K1, bs, s0, s1, w0, w1, obj_geod_th=1e-5
):

    bs_dist = symmetric_epipolar_distance(
        bs_kpts0, bs_kpts1, E_0to1[bs], K0[bs], K1[bs]
    )
    bs_dist_mask = bs_dist <= obj_geod_th
    del bs_dist
    bs_kpts0, bs_kpts1 = bs_kpts0[bs_dist_mask], bs_kpts1[bs_dist_mask]
    # b_ids0, i_ids0, j_ids0
    bs_grid_pt0 = bs_kpts0
    bs_w_pt0 = bs_kpts1
    bs_grid_pt0_c = (bs_kpts0 / s0[bs]).round().long()
    bs_w_pt0_c_long = (bs_kpts1 / s1[bs]).round().long()
    f_bs_i_ids0 = bs_grid_pt0_c[:, 0] + bs_grid_pt0_c[:, 1] * w0
    f_bs_j_ids0 = bs_w_pt0_c_long[:, 0] + bs_w_pt0_c_long[:, 1] * w1
    f_bs_b_ids0 = torch.full_like(f_bs_j_ids0, bs)

    # b_ids1, j_ids1, i_ids1
    f_bs_j_ids1 = (bs_w_pt0_c_long[:, 0] + bs_w_pt0_c_long[:, 1] * w1).unique()
    bs_grid_pt1_c = torch.stack([f_bs_j_ids1 % w1, f_bs_j_ids1 // w1], dim=1)
    bs_grid_pt1 = bs_grid_pt1_c * s1[bs]
    n1 = len(bs_grid_pt1)
    n0 = len(bs_kpts0)
    tomatch_pt1 = rearrange(
        bs_grid_pt1.unsqueeze(1).repeat(1, n0, 1), "n1 n0 c -> (n1 n0) c"
    )
    tomatch_pt0 = rearrange(bs_kpts0.repeat(n1, 1, 1), "n1 n0 c -> (n1 n0) c")
    match_scores = symmetric_epipolar_distance(
        tomatch_pt1, tomatch_pt0, E_1to0[bs], K1[bs], K0[bs]
    )
    v, ind = rearrange(match_scores, "(n1 n0) -> n1 n0", n1=n1, n0=n0).min(dim=1)
    del match_scores, tomatch_pt1, tomatch_pt0
    bs_w_pt1 = bs_kpts0[ind]
    bs_w_pt1_c_long = (bs_w_pt1 / s0[bs]).round().long()
    f_bs_i_ids1 = bs_w_pt1_c_long[:, 0] + bs_w_pt1_c_long[:, 1] * w0
    f_bs_b_ids1 = torch.full_like(f_bs_i_ids1, bs)

    return (
        bs_grid_pt0,
        bs_w_pt0,
        f_bs_b_ids0,
        f_bs_i_ids0,
        f_bs_j_ids0,
        bs_grid_pt1,
        bs_w_pt1,
        f_bs_b_ids1,
        f_bs_j_ids1,
        f_bs_i_ids1,
    )


@torch.no_grad()
def get_scale_gt_matrix5(
    data, scale_l, N, H0, W0, H1, W1, device, require_depth_mask=True, obj_geod_th=1e-5
):
    # pdb.set_trace()
    device = data["K0"].device
    bs = data["K0"].shape[0]
    scale0 = (
        scale_l * data["scale0"][:, None]
        if "scale0" in data
        else torch.tensor(
            [[[scale_l, scale_l]]], dtype=torch.float, device=device
        ).repeat(bs, 1, 1)
    )  # float(scale_l)
    scale1 = (
        scale_l * data["scale1"][:, None]
        if "scale0" in data
        else torch.tensor(
            [[[scale_l, scale_l]]], dtype=torch.float, device=device
        ).repeat(bs, 1, 1)
    )  # float(scale_l)
    h0, w0, h1, w1 = map(lambda x: x // scale_l, [H0, W0, H1, W1])
    scale_wh0_l, scale_wh1_l = (data["scale_wh0"] // scale_l), (
        data["scale_wh1"] // scale_l
    )

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)
    )  # [N, hw, 2] - <x, y>
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
    )
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0_d{}".format(int(scale_l)) in data:
        grid_pt0_i = mask_pts_at_padded_regions(
            grid_pt0_i, data["mask0_d{}".format(int(scale_l))]
        )  # [N, L=h0*w0, 2] - <x, y>
        grid_pt1_i = mask_pts_at_padded_regions(
            grid_pt1_i, data["mask1_d{}".format(int(scale_l))]
        )  # [N, S=h1*w1, 2] - <x, y>

    valid_mask0, w_pt0_i, d_mask0, consistent_mask0 = warp_kpts(
        grid_pt0_i,
        data["depth0"],
        data["depth1"],
        data["T_0to1"],
        data["T_1to0"],
        data["K0"],
        data["K1"],
    )  # 原图尺寸
    valid_mask1, w_pt1_i, d_mask1, consistent_mask1 = warp_kpts(
        grid_pt1_i,
        data["depth1"],
        data["depth0"],
        data["T_1to0"],
        data["T_0to1"],
        data["K1"],
        data["K0"],
    )
    # s_wh1 = scale_wh1_l.repeat(1, valid_mask0.shape[1], 1)
    s_wh1 = scale_wh1_l.unsqueeze(1).repeat(1, valid_mask0.shape[1], 1)
    w_pt0_all_long = (w_pt0_i / scale1).round().long()
    covisible_mask0 = (
        (w_pt0_all_long[..., 0] > 0)
        * (w_pt0_all_long[..., 0] < s_wh1[..., 0] - 1)
        * (w_pt0_all_long[..., 1] > 0)
        * (w_pt0_all_long[..., 1] < s_wh1[..., 1] - 1)
    )
    # s_wh0 = scale_wh0_l.repeat(1, valid_mask1.shape[1], 1)
    s_wh0 = scale_wh0_l.unsqueeze(1).repeat(1, valid_mask1.shape[1], 1)
    w_pt1_all_long = (w_pt1_i / scale0).round().long()
    covisible_mask1 = (
        (w_pt1_all_long[..., 0] > 0)
        * (w_pt1_all_long[..., 0] < s_wh0[..., 0] - 1)
        * (w_pt1_all_long[..., 1] > 0)
        * (w_pt1_all_long[..., 1] < s_wh0[..., 1] - 1)
    )
    flag = 0
    if (valid_mask0 * covisible_mask0 * consistent_mask0 == 0).all() and (
        valid_mask1 * covisible_mask1 * consistent_mask1 == 0
    ).all():
        flag = 1
        valid_mask0 = valid_mask0 * covisible_mask0
        valid_mask1 = valid_mask1 * covisible_mask1
    else:
        valid_mask0 = valid_mask0 * consistent_mask0 * covisible_mask0
        valid_mask1 = valid_mask1 * consistent_mask1 * covisible_mask1
    del covisible_mask0, covisible_mask1, consistent_mask0, consistent_mask1

    b_ids0, i_ids0 = torch.nonzero(valid_mask0, as_tuple=True)
    v_w_pt0_i_long = w_pt0_all_long[
        b_ids0, i_ids0
    ]  # (w_pt0_i / scale1)[b_ids0, i_ids0].round().long()
    j_ids0 = v_w_pt0_i_long[:, 0] + v_w_pt0_i_long[:, 1] * w1
    # s_wh1 = scale_wh1_l[b_ids0]
    # covisible_mask0 = (v_w_pt0_i_long[:, 0] > 0) * (v_w_pt0_i_long[:, 0] < s_wh1[:, 0] - 1) * \
    #                   (v_w_pt0_i_long[:, 1] > 0) * (v_w_pt0_i_long[:, 1] < s_wh1[:, 1] - 1)
    # b_ids0, i_ids0 = b_ids0[covisible_mask0], i_ids0[covisible_mask0]
    # j_ids0 = v_w_pt0_i_long[covisible_mask0][:, 0] + v_w_pt0_i_long[covisible_mask0][:, 1] * w1

    b_ids1, j_ids1 = torch.nonzero(valid_mask1, as_tuple=True)
    v_w_pt1_i_long = w_pt1_all_long[
        b_ids1, j_ids1
    ]  # (w_pt1_i / scale0)[b_ids1, j_ids1].round().long()
    i_ids1 = v_w_pt1_i_long[:, 0] + v_w_pt1_i_long[:, 1] * w0
    # s_wh0 = scale_wh0_l[b_ids1]
    # covisible_mask1 = (v_w_pt1_i_long[:, 0] > 0) * (v_w_pt1_i_long[:, 0] < s_wh0[:, 0] - 1) * \
    #                   (v_w_pt1_i_long[:, 1] > 0) * (v_w_pt1_i_long[:, 1] < s_wh0[:, 1] - 1)
    # b_ids1, j_ids1 = b_ids1[covisible_mask1], j_ids1[covisible_mask1]
    # i_ids1 = v_w_pt1_i_long[covisible_mask1][:, 0] + v_w_pt1_i_long[covisible_mask1][:, 1] * w0

    f_b_ids0, f_i_ids0, f_j_ids0, f_b_ids1, f_j_ids1, f_i_ids1, = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    f_grid_pt0, f_w_pt0, f_grid_pt1, f_w_pt1 = [], [], [], []
    E_0to1, F_0to1 = pose2essential_fundamental(data["K0"], data["K1"], data["T_0to1"])
    E_1to0, F_1to0 = pose2essential_fundamental(data["K1"], data["K0"], data["T_1to0"])

    for bs in range(N):
        bs_mask0 = b_ids0 == bs
        bs_b_ids0, bs_i_ids0, bs_j_ids0 = (
            b_ids0[bs_mask0],
            i_ids0[bs_mask0],
            j_ids0[bs_mask0],
        )
        bs_mask1 = b_ids1 == bs
        bs_b_ids1, bs_j_ids1, bs_i_ids1 = (
            b_ids1[bs_mask1],
            j_ids1[bs_mask1],
            i_ids1[bs_mask1],
        )

        if len(bs_i_ids0) == 0 or len(bs_j_ids1) == 0:
            try:
                if len(bs_i_ids0) > len(bs_j_ids1):
                    bs_kpts0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
                    bs_kpts1 = w_pt0_i[bs_b_ids0, bs_i_ids0]
                    (
                        bs_grid_pt0,
                        bs_w_pt0,
                        f_bs_b_ids0,
                        f_bs_i_ids0,
                        f_bs_j_ids0,
                        bs_grid_pt1,
                        bs_w_pt1,
                        f_bs_b_ids1,
                        f_bs_j_ids1,
                        f_bs_i_ids1,
                    ) = get_warp_index(
                        bs_kpts0,
                        bs_kpts1,
                        E_0to1,
                        E_1to0,
                        data["K0"],
                        data["K1"],
                        bs,
                        scale0,
                        scale1,
                        w0,
                        w1,
                        obj_geod_th,
                    )
                else:
                    bs_kpts1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
                    bs_kpts0 = w_pt1_i[bs_b_ids1, bs_j_ids1]
                    (
                        bs_grid_pt1,
                        bs_w_pt1,
                        f_bs_b_ids1,
                        f_bs_j_ids1,
                        f_bs_i_ids1,
                        bs_grid_pt0,
                        bs_w_pt0,
                        f_bs_b_ids0,
                        f_bs_i_ids0,
                        f_bs_j_ids0,
                    ) = get_warp_index(
                        bs_kpts1,
                        bs_kpts0,
                        E_1to0,
                        E_0to1,
                        data["K1"],
                        data["K0"],
                        bs,
                        scale1,
                        scale0,
                        w1,
                        w0,
                        obj_geod_th,
                    )
            except:
                print(data["scene_id"], data["pair_id"], data["pair_names"])
                bs_grid_pt0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
                bs_w_pt0 = w_pt0_i[bs_b_ids0, bs_i_ids0]
                f_bs_b_ids0, f_bs_i_ids0, f_bs_j_ids0 = bs_b_ids0, bs_i_ids0, bs_j_ids0
                bs_grid_pt1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
                bs_w_pt1 = w_pt1_i[bs_b_ids1, bs_j_ids1]
                f_bs_b_ids1, f_bs_j_ids1, f_bs_i_ids1 = bs_b_ids1, bs_j_ids1, bs_i_ids1
        else:
            bs_grid_pt0 = grid_pt0_i[bs_b_ids0, bs_i_ids0]
            bs_w_pt0 = w_pt0_i[bs_b_ids0, bs_i_ids0]
            f_bs_b_ids0, f_bs_i_ids0, f_bs_j_ids0 = bs_b_ids0, bs_i_ids0, bs_j_ids0

            bs_grid_pt1 = grid_pt1_i[bs_b_ids1, bs_j_ids1]
            bs_w_pt1 = w_pt1_i[bs_b_ids1, bs_j_ids1]
            f_bs_b_ids1, f_bs_j_ids1, f_bs_i_ids1 = bs_b_ids1, bs_j_ids1, bs_i_ids1

        geod_mask0 = (
            symmetric_epipolar_distance(
                bs_grid_pt0, bs_w_pt0, E_0to1[bs], data["K0"][bs], data["K1"][bs]
            )
            <= obj_geod_th
        )
        geod_mask1 = (
            symmetric_epipolar_distance(
                bs_w_pt1, bs_grid_pt1, E_0to1[bs], data["K0"][bs], data["K1"][bs]
            )
            <= obj_geod_th
        )

        f_b_ids0.append(f_bs_b_ids0[geod_mask0])
        f_i_ids0.append(f_bs_i_ids0[geod_mask0])
        f_j_ids0.append(f_bs_j_ids0[geod_mask0])
        f_grid_pt0.append(bs_grid_pt0[geod_mask0])
        f_w_pt0.append(bs_w_pt0[geod_mask0])

        f_b_ids1.append(f_bs_b_ids1[geod_mask1])
        f_j_ids1.append(f_bs_j_ids1[geod_mask1])
        f_i_ids1.append(f_bs_i_ids1[geod_mask1])
        f_grid_pt1.append(bs_grid_pt1[geod_mask1])
        f_w_pt1.append(bs_w_pt1[geod_mask1])

    f_b_ids0 = torch.cat(f_b_ids0, dim=0)
    f_i_ids0 = torch.cat(f_i_ids0, dim=0)
    f_j_ids0 = torch.cat(f_j_ids0, dim=0)
    f_grid_pt0_i = torch.cat(f_grid_pt0, dim=0)
    f_w_pt0_i = torch.cat(f_w_pt0, dim=0)

    f_b_ids1 = torch.cat(f_b_ids1, dim=0)
    f_j_ids1 = torch.cat(f_j_ids1, dim=0)
    f_i_ids1 = torch.cat(f_i_ids1, dim=0)
    f_grid_pt1_i = torch.cat(f_grid_pt1, dim=0)
    f_w_pt1_i = torch.cat(f_w_pt1, dim=0)

    w_pt0_i[f_b_ids0, f_i_ids0] = f_w_pt0_i
    w_pt1_i[f_b_ids1, f_j_ids1] = f_w_pt1_i

    if require_depth_mask:
        # pdb.set_trace()
        depth_mask0 = torch.zeros_like(d_mask0).bool()
        depth_mask0[f_b_ids0, f_i_ids0] = True
        depth_mask1 = torch.zeros_like(d_mask1).bool()
        depth_mask1[f_b_ids1, f_j_ids1] = True

        return (
            f_b_ids0,
            f_i_ids0,
            f_j_ids0,
            f_b_ids1,
            f_j_ids1,
            f_i_ids1,
            depth_mask0,
            depth_mask1,
            w_pt0_i,
            grid_pt0_i,
            w_pt1_i,
            grid_pt1_i,
        )
        # f_w_pt0_i, f_grid_pt0_i, \
        # f_w_pt1_i, f_grid_pt1_i
    else:
        return (
            f_b_ids0,
            f_i_ids0,
            f_j_ids0,
            f_b_ids1,
            f_j_ids1,
            f_i_ids1,
            w_pt0_i,
            grid_pt0_i,
            w_pt1_i,
            grid_pt1_i,
        )
        # f_w_pt0_i, f_grid_pt0_i, \
        # f_w_pt1_i, f_grid_pt1_i


@torch.no_grad()
def spvs_instance_gt2(data, config):
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale_l0, scale_l1, scale_l2 = config["ADAMATCHER"]["RESOLUTION"]  # 64, 8, 2
    scale_l0l1 = scale_l0 // scale_l1
    # scale_l1l2 = scale_l1 // scale_l2
    h0_l0, w0_l0, h1_l0, w1_l0 = map(lambda x: x // scale_l0, [H0, W0, H1, W1])
    h0_l1, w0_l1, h1_l1, w1_l1 = map(lambda x: x // scale_l1, [H0, W0, H1, W1])
    h0_l2, w0_l2, h1_l2, w1_l2 = map(lambda x: x // scale_l2, [H0, W0, H1, W1])

    ####################
    # b_ids0_l1, i_ids0_l1, j_ids0_l1, \
    # b_ids1_l1, j_ids1_l1, i_ids1_l1, \
    # depth_mask0_l1, depth_mask1_l1, \
    # w_pt0_i_l1, grid_pt0_i_l1, \
    # w_pt1_i_l1, grid_pt1_i_l1 \
    #     = get_scale_gt_matrix4(data, scale_l1, N, H0, W0, H1, W1,
    #                            device, require_depth_mask=True, obj_geod_th=1e-4)
    (
        b_ids0_l2,
        i_ids0_l2,
        j_ids0_l2,
        b_ids1_l2,
        j_ids1_l2,
        i_ids1_l2,
        depth_mask0_l2,
        depth_mask1_l2,
        w_pt0_i_l2,
        grid_pt0_i_l2,
        w_pt1_i_l2,
        grid_pt1_i_l2,
    ) = get_scale_gt_matrix5(
        data,
        scale_l2,
        N,
        H0,
        W0,
        H1,
        W1,
        device,
        require_depth_mask=True,
        obj_geod_th=1e-4,
    )
    flag = 0
    import pdb

    # pdb.set_trace()
    if flag:
        import os
        import pdb
        import pickle

        import numpy as np

        pdb.set_trace()
        simg0 = (
            (data["image0"][0].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )
        simg1 = (
            (data["image1"][0].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )
        sdepth0 = data["depth0"][0].cpu().numpy()
        sdepth1 = data["depth1"][0].cpu().numpy()
        sw_pt0_i_l2 = w_pt0_i_l2[0].cpu().numpy()
        sgrid_pt0_i_l2 = grid_pt0_i_l2[0].cpu().numpy()
        sw_pt1_i_l2 = w_pt1_i_l2[0].cpu().numpy()
        sgrid_pt1_i_l2 = grid_pt1_i_l2[0].cpu().numpy()
        sdepth_mask0_l2 = depth_mask0_l2[0].cpu().numpy()
        sdepth_mask1_l2 = depth_mask1_l2[0].cpu().numpy()
        save_dict = dict(
            img0=simg0,
            img1=simg1,
            depth0=sdepth0,
            depth1=sdepth1,
            w_pt0_i_l2=sw_pt0_i_l2,
            grid_pt0_i_l2=sgrid_pt0_i_l2,
            w_pt1_i_l2=sw_pt1_i_l2,
            grid_pt1_i_l2=sgrid_pt1_i_l2,
            depth_mask0_l2=sdepth_mask0_l2,
            depth_mask1_l2=sdepth_mask1_l2,
            scale0=data["scale0"][0].cpu().numpy(),
            scale1=data["scale1"][0].cpu().numpy(),
        )
        pkl_path = "./viz_pkl/2.pkl"
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        pkl_file = open(pkl_path, "wb")
        pickle.dump(save_dict, pkl_file)
        pkl_file.close()
    depth_mask0_l2 = rearrange(depth_mask0_l2, "n (h w) -> n h w", h=h0_l2, w=w0_l2)
    depth_mask1_l2 = rearrange(depth_mask1_l2, "n (h w) -> n h w", h=h1_l2, w=w1_l2)
    depth_mask0_l1 = (
        rearrange(
            depth_mask0_l2, "n (s_h h) (s_w w) -> n s_h s_w (h w)", s_h=h0_l1, s_w=w0_l1
        ).sum(dim=-1)
        > 0
    )  # / (scale_l1l2**2) > 0.1
    depth_mask1_l1 = (
        rearrange(
            depth_mask1_l2, "n (s_h h) (s_w w) -> n s_h s_w (h w)", s_h=h1_l1, s_w=w1_l1
        ).sum(dim=-1)
        > 0
    )  # / (scale_l1l2**2) > 0.1

    (
        b_ids0_l1,
        i_ids0_l1,
        j_ids0_l1,
        b_ids1_l1,
        j_ids1_l1,
        i_ids1_l1,
        w_pt0_i_l1,
        grid_pt0_i_l1,
        w_pt1_i_l1,
        grid_pt1_i_l1,
    ) = get_scale_gt_matrix5(
        data,
        scale_l1,
        N,
        H0,
        W0,
        H1,
        W1,
        device,
        require_depth_mask=False,
        obj_geod_th=1e-4,
    )
    # get_scale_gt_matrix3

    # img0: patch(sub pixel),  img1: mask(grid)
    valid_kpts0_l1 = torch.stack([i_ids1_l1 % w0_l1, i_ids1_l1 // w0_l1], dim=1)
    valid_kpts0_l0 = valid_kpts0_l1 // scale_l0l1
    valid_kpts0_l0l1 = valid_kpts0_l1 - valid_kpts0_l0 * scale_l0l1
    assert (valid_kpts0_l0l1 >= 0).all()
    i_ids1_l0 = valid_kpts0_l0[:, 0] + valid_kpts0_l0[:, 1] * w0_l0
    i_ids1_l0l1 = valid_kpts0_l0l1[:, 0] + valid_kpts0_l0l1[:, 1] * scale_l0l1
    assert (
        (
            (i_ids1_l0 % w0_l0 * scale_l0l1 + i_ids1_l0l1 % scale_l0l1)
            + (i_ids1_l0 // w0_l0 * scale_l0l1 + i_ids1_l0l1 // scale_l0l1) * w0_l1
        )
        == i_ids1_l1
    ).all()
    data.update(
        {
            "spv_b_ids1_l1": b_ids1_l1,
            "spv_j_ids1_l1": j_ids1_l1,
            "spv_i_ids1_l1": i_ids1_l1,
            "spv_i_ids1_l0": i_ids1_l0,
            "spv_i_ids1_l0l1": i_ids1_l0l1,
            "spv_w_pt1_i_l1": w_pt1_i_l1,
            "spv_pt1_i_l1": grid_pt1_i_l1,
        }
    )

    # img0: mask(grid), img1: patch(sub pixel)
    valid_kpts1_l1 = torch.stack([j_ids0_l1 % w1_l1, j_ids0_l1 // w1_l1], dim=1)
    valid_kpts1_l0 = valid_kpts1_l1 // scale_l0l1
    valud_kpts1_l0l1 = valid_kpts1_l1 - valid_kpts1_l0 * scale_l0l1
    assert (valud_kpts1_l0l1 >= 0).all()
    j_ids0_l0 = valid_kpts1_l0[:, 0] + valid_kpts1_l0[:, 1] * w1_l0
    j_ids0_l0l1 = valud_kpts1_l0l1[:, 0] + valud_kpts1_l0l1[:, 1] * scale_l0l1
    assert (
        (
            (j_ids0_l0 % w1_l0 * scale_l0l1 + j_ids0_l0l1 % scale_l0l1)
            + (j_ids0_l0 // w1_l0 * scale_l0l1 + j_ids0_l0l1 // scale_l0l1) * w1_l1
        )
        == j_ids0_l1
    ).all()
    data.update(
        {
            "spv_b_ids0_l1": b_ids0_l1,
            "spv_i_ids0_l1": i_ids0_l1,
            "spv_j_ids0_l1": j_ids0_l1,
            "spv_j_ids0_l0": j_ids0_l0,
            "spv_j_ids0_l0l1": j_ids0_l0l1,
            "spv_w_pt0_i_l1": w_pt0_i_l1,
            "spv_pt0_i_l1": grid_pt0_i_l1,
        }
    )

    # construct gt class matrix  dm_: depth mask
    # class_matrix_l1_gt0 = rearrange(depth_mask0_l1, 'n (h w) -> n h w', h=h0_l1, w=w0_l1).float()
    # class_matrix_l1_gt1 = rearrange(depth_mask1_l1, 'n (h w) -> n h w', h=h1_l1, w=w1_l1).float()
    class_matrix_l1_gt0 = depth_mask0_l1.float()
    class_matrix_l1_gt1 = depth_mask1_l1.float()

    # class matrix l0
    class_matrix_l0_gt0 = torch.zeros(
        (N, h0_l0, w0_l0), device=device
    )  # 1 for positive  c0 background c1 front
    class_matrix_l0_gt1 = torch.zeros((N, h1_l0, w1_l0), device=device)

    dnum_matrix_l0_gt0 = (
        (
            rearrange(
                class_matrix_l1_gt0,
                "n (s_h h) (s_w w) -> n s_h s_w (h w)",
                s_h=h0_l0,
                s_w=w0_l0,
            )
            > 0
        )
        .sum(dim=3)
        .float()
    )
    dnum_matrix_l0_gt0 = dnum_matrix_l0_gt0 / (
        (h0_l1 * w0_l1) / (h0_l0 * w0_l0)
    )  # (dnum_matrix_l0_gt0.size(1) * dnum_matrix_l0_gt0.size(2))
    class_matrix_l0_gt0[
        torch.nonzero(dnum_matrix_l0_gt0 > 0.1, as_tuple=True)
    ] = 1  # [N, h0_l0, w0_l0] # 0.2
    class_matrix_l0_gt0 = class_matrix_l0_gt0.unsqueeze(1)  # [N, 1, h0_l0, w0_l0]
    class_matrix_l0_gt0 = torch.cat(
        [1 - class_matrix_l0_gt0, class_matrix_l0_gt0], dim=1
    )  # [N, 2, h0_l0, w0_l0]
    class_b0_l0_ids, class_k0_l0_ids = torch.nonzero(
        rearrange(class_matrix_l0_gt0[:, 1, :, :], "n h w -> n (h w)"), as_tuple=True
    )

    dnum_matrix_l0_gt1 = (
        (
            rearrange(
                class_matrix_l1_gt1,
                "n (s_h h) (s_w w) -> n s_h s_w (h w)",
                s_h=h1_l0,
                s_w=w1_l0,
            )
            > 0
        )
        .sum(dim=3)
        .float()
    )
    dnum_matrix_l0_gt1 = dnum_matrix_l0_gt1 / (
        (h1_l1 * w1_l1) / (h1_l0 * w1_l0)
    )  # (dnum_matrix_l0_gt1.size(1) * dnum_matrix_l0_gt1.size(2))
    class_matrix_l0_gt1[
        torch.nonzero(dnum_matrix_l0_gt1 > 0.1, as_tuple=True)
    ] = 1  # [N, h1_l0, w1_l0] # 0.2
    class_matrix_l0_gt1 = class_matrix_l0_gt1.unsqueeze(1)  # [N, 1, h1_l0, w1_l0]
    class_matrix_l0_gt1 = torch.cat(
        [1 - class_matrix_l0_gt1, class_matrix_l0_gt1], dim=1
    )  # [N, 2, h1_l0, w1_l0]
    class_b1_l0_ids, class_k1_l0_ids = torch.nonzero(
        rearrange(class_matrix_l0_gt1[:, 1, :, :], "n h w -> n (h w)"), as_tuple=True
    )

    # class matrix l1
    class_matrix_l1_gt0 = class_matrix_l1_gt0  # .unsqueeze(1)  # [N, 1, h0_l1, w0_l1]
    # class_matrix_l1_gt0 = torch.cat([1 - class_matrix_l1_gt0, class_matrix_l1_gt0], dim=1)  # [N, 2, h0_l1, w0_l1]

    class_matrix_l1_gt1 = class_matrix_l1_gt1  # .unsqueeze(1)  # [N, 1, h1_l1, w1_l1]
    # class_matrix_l1_gt1 = torch.cat([1 - class_matrix_l1_gt1, class_matrix_l1_gt1], dim=1)  # [N, 2, h1_l1, w1_l1]

    data.update(
        {
            "spv_class_b0_l0_ids": class_b0_l0_ids,  # [k,]
            "spv_class_k0_l0_ids": class_k0_l0_ids,  # [k,]
            "spv_class_b1_l0_ids": class_b1_l0_ids,  # [k,]
            "spv_class_k1_l0_ids": class_k1_l0_ids,  # [k,]
            "spv_class_l1_gt0": class_matrix_l1_gt0,  # [N, 2, h0_l1, w0_l1]
            "spv_class_l1_gt1": class_matrix_l1_gt1,  # [N, 2, h1_l1, w1_l1]
        }
    )

    # construct gt instance mask
    # conf_matrix1_l1  img0: patch(sub pixel)  img1: mask
    conf_matrix1_l1 = torch.zeros((N, h0_l1 * w0_l1, h1_l1 * w1_l1), device=device)
    conf_matrix1_l1[b_ids1_l1, i_ids1_l1, j_ids1_l1] = 1
    # b_ids1_l1, j_ids1_l1, i_ids1_l1
    # mid_matrix1_l1 = rearrange(conf_matrix1_l1,
    #                            'n (h0 ph0 w0 pw0) (h1 w1) -> n (h0 w0) (ph0 pw0) (h1 w1)',
    #                            h0=h0_l0, w0=w0_l0, ph0=h0_l1 // h0_l0, pw0=w0_l1 // w0_l0,
    #                            h1=h1_l1, w1=w1_l1)
    # instance_masks1_l1 = mid_matrix1_l1.sum(dim=2)  # [n, (h0 w0), (h1, w1) ]  # .bool().float()  TODO (h0 w0) (ph0 pw0)
    instance_masks1_l1 = rearrange(
        conf_matrix1_l1,
        "n (h0 ph0 w0 pw0) (h1 w1) -> n (h0 w0) (ph0 pw0) (h1 w1)",
        h0=h0_l0,
        w0=w0_l0,
        ph0=h0_l1 // h0_l0,
        pw0=w0_l1 // w0_l0,
        h1=h1_l1,
        w1=w1_l1,
    ).sum(dim=2)
    instance_masks1_l1 = rearrange(
        instance_masks1_l1, "n l (h1 w1) -> n l h1 w1", h1=h1_l1, w1=w1_l1
    ).float()

    ######################################
    conf_matrix0_l1 = torch.zeros((N, h1_l1 * w1_l1, h0_l1 * w0_l1), device=device)
    conf_matrix0_l1[b_ids0_l1, j_ids0_l1, i_ids0_l1] = 1
    # b_ids0_l1, i_ids0_l1, j_ids0_l1
    # mid_matrix0_l1 = rearrange(conf_matrix0_l1,
    #                            'n (h0 ph0 w0 pw0) (h1 w1) -> n (h0 w0) (ph0 pw0) (h1 w1)',
    #                            h0=h1_l0, w0=w1_l0, ph0=h1_l1 // h1_l0, pw0=w1_l1 // w1_l0,
    #                            h1=h0_l1, w1=w0_l1)
    # instance_masks0_l1 = mid_matrix0_l1.sum(dim=2)  # .bool().float()
    instance_masks0_l1 = rearrange(
        conf_matrix0_l1,
        "n (h0 ph0 w0 pw0) (h1 w1) -> n (h0 w0) (ph0 pw0) (h1 w1)",
        h0=h1_l0,
        w0=w1_l0,
        ph0=h1_l1 // h1_l0,
        pw0=w1_l1 // w1_l0,
        h1=h0_l1,
        w1=w0_l1,
    ).sum(dim=2)
    instance_masks0_l1 = rearrange(
        instance_masks0_l1, "n l (h1 w1) -> n l h1 w1", h1=h0_l1, w1=w0_l1
    ).float()

    data.update(
        {
            "spv_instance_masks0": instance_masks0_l1,  # [N, h1_l0*w1_l0, h0_l1, w0_l1]
            "spv_conf_matrix0_l1": conf_matrix0_l1,  # [N, h1_l1*w1_l1, h0_l1*w0_l1]
            # 'spv_mid_matrix0_l0l1': mid_matrix0_l1,  # [N, h1_l0*w1_l0, 64, h0_l1*w0_l1]
            "spv_instance_masks1": instance_masks1_l1,  # [N, h0_l0*w0_l0, h1_l1, w1_l1]
            "spv_conf_matrix1_l1": conf_matrix1_l1,  # [N, h0_l1*w0_l1, h1_l1*w1_l1]
            # 'spv_mid_matrix1_l0l1': mid_matrix1_l1,  # [N, h0_l0*w0_l0, 64, h1_l1*w1_l1]
        }
    )


def compute_supervision_coarse(data, config):
    assert (
        len(set(data["dataset_name"])) == 1
    ), "Do not support mixed datasets training!"
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        # spvs_coarse(data, config)
        spvs_instance_gt2(data, config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


##############  ↓  Fine-Level supervision  ↓  ##############


@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i_l2, pt0_i_l2 = data["spv_w_pt0_i_l2"], data["spv_pt0_i_l2"]
    w_pt1_i_l2, pt1_i_l2 = data["spv_w_pt1_i_l2"], data["spv_pt1_i_l2"]
    scale_l0, scale_l1, scale_l2 = config["ADAMATCHER"]["RESOLUTION"]  # 64, 8, 2
    radius = config["ADAMATCHER"]["FINE_WINDOW_SIZE"] // 2

    # 2. get coarse prediction
    b_ids_l2, i_ids_l2, j_ids_l2 = data["b_ids_l2"], data["i_ids_l2"], data["j_ids_l2"]
    gt_w_pt0_l2 = w_pt0_i_l2[b_ids_l2, i_ids_l2]
    gt_pt0_l2 = pt0_i_l2[b_ids_l2, i_ids_l2]

    gt_w_pt1_l2 = w_pt1_i_l2[b_ids_l2, j_ids_l2]
    gt_pt1_l2 = pt1_i_l2[b_ids_l2, j_ids_l2]

    # 3. compute gt
    scale = scale_l2 * data["scale1"][b_ids_l2] if "scale0" in data else scale_l2
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (
        (w_pt0_i_l2[b_ids_l2, i_ids_l2] - pt1_i_l2[b_ids_l2, j_ids_l2]) / scale / radius
    )  # [M, 2]
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
