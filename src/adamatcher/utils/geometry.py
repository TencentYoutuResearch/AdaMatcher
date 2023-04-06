import pdb

import torch
from kornia.geometry.epipolar import essential, fundamental, numeric


@torch.no_grad()
def skew(v):
    # The skew-symmetric matrix of vector
    return torch.tensor(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=v.device
    )


@torch.no_grad()
def pose2fundamental(K0, K1, T_0to1):
    # pdb.set_trace()
    Tx = numeric.cross_product_matrix(T_0to1[:, :3, 3])
    E_mat = Tx @ T_0to1[:, :3, :3]
    # F = torch.inverse(K1).T @ R0to1 @ K0.T @ skew((K0 @ R0to1.T).dot(t0to1.reshape(3,)))
    # F_mat = torch.inverse(K1).T @ E_mat @ torch.inverse(K0)
    # F_mat = fundamental.fundamental_from_essential(E_mat, K0, K1)
    F_mat = torch.inverse(K1).transpose(1, 2) @ E_mat @ torch.inverse(K0)
    return F_mat


@torch.no_grad()
def pose2essential_fundamental(K0, K1, T_0to1):
    # pdb.set_trace()
    Tx = numeric.cross_product_matrix(T_0to1[:, :3, 3])
    E_mat = Tx @ T_0to1[:, :3, :3]
    F_mat = torch.inverse(K1).transpose(1, 2) @ E_mat @ torch.inverse(K0)
    return E_mat, F_mat


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1=None, T_1to0=None, K0=None, K1=None):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W], depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3], K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
        depth_mask
    """
    # kpts0_depth = interpolate_depth(kpts0, depth0)
    # pdb.set_trace()
    kpts0_long = kpts0.round().long()
    kpts0_depth = torch.stack(
        [
            depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
            for i in range(kpts0.shape[0])
        ],
        dim=0,
    )  # (N, L)
    depth_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0
    # w_kpts0_depth = interpolate_depth(w_kpts0, depth1)
    w_kpts0_long = w_kpts0.round().long()
    w_kpts0_long[~covisible_mask, :] = 0
    w_kpts0_depth = torch.stack(
        [
            depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]]
            for i in range(w_kpts0.shape[0])
        ],
        dim=0,
    )
    if T_1to0 is None:
        consistent_mask = (
            (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
        ).abs() < 0.2  # 0.2
        valid_mask = depth_mask * covisible_mask * consistent_mask

        return valid_mask, w_kpts0, depth_mask
    else:
        kpts1_h = (
            torch.cat([w_kpts0, torch.ones_like(w_kpts0[:, :, [0]])], dim=-1)
            * w_kpts0_depth[..., None]
        )
        kpts1_cam = K1.inverse() @ kpts1_h.transpose(2, 1)  # (N, 3, L)
        w_kpts1_cam = T_1to0[:, :3, :3] @ kpts1_cam + T_1to0[:, :3, [3]]
        w_kpts1_h = (K0 @ w_kpts1_cam).transpose(2, 1)  # (N, L, 3)
        w_kpts1 = w_kpts1_h[:, :, :2] / (w_kpts1_h[:, :, [2]] + 1e-4)  # (N, L, 2)
        consistent_mask = torch.norm(w_kpts1 - kpts0, p=2, dim=-1) < 4.0  # 4.  5.

        # pdb.set_trace()
        # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.5 # 0.2  # 0.2
        valid_mask = depth_mask * covisible_mask  # * consistent_mask

        return valid_mask, w_kpts0, depth_mask, consistent_mask


@torch.no_grad()
def interpolate_depth(position, depth):
    """
    Args:
        position: [N, l, 2(x,y)]  # [2(y,x), m]
        depth: [N, H, W]
    output:
        interpolated_depth: [m']
        position: [N, m', 2(x,y)] # [2(y,x), m']
        ids: [m']
    """
    N, H, W = depth.size()
    # ids = torch.arange(0, position.size(1)).repeat(N, 1)
    kpts_depth = torch.zeros_like(position[:, :, 0])  # [N, l]

    i = position[:, :, 1]  # y [N, l]
    j = position[:, :, 0]  # x [N, l]

    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < W)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < H, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < H, j_bottom_right < W)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right),
    )  # [N, l]

    b_ids, k_ids = torch.nonzero(valid_corners, as_tuple=True)

    i_top_left = i_top_left[b_ids, k_ids]
    j_top_left = j_top_left[b_ids, k_ids]

    i_top_right = i_top_right[b_ids, k_ids]
    j_top_right = j_top_right[b_ids, k_ids]

    i_bottom_left = i_bottom_left[b_ids, k_ids]
    j_bottom_left = j_bottom_left[b_ids, k_ids]

    i_bottom_right = i_bottom_right[b_ids, k_ids]
    j_bottom_right = j_bottom_right[b_ids, k_ids]

    # ids = ids[valid_corners]
    if len(k_ids) == 0:
        return kpts_depth

    # Valid depth
    valid_depth = torch.min(
        torch.min(
            depth[b_ids, i_top_left, j_top_left] > 0,
            depth[b_ids, i_top_right, j_top_right] > 0,
        ),
        torch.min(
            depth[b_ids, i_bottom_left, j_bottom_left] > 0,
            depth[b_ids, i_bottom_right, j_bottom_right] > 0,
        ),
    )

    b_ids = b_ids[valid_depth]
    k_ids = k_ids[valid_depth]

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    if len(k_ids) == 0:
        return kpts_depth

    i = i[b_ids, k_ids]
    j = j[b_ids, k_ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    interpolated_depth = (
        w_top_left * depth[b_ids, i_top_left, j_top_left]
        + w_top_right * depth[b_ids, i_top_right, j_top_right]
        + w_bottom_left * depth[b_ids, i_bottom_left, j_bottom_left]
        + w_bottom_right * depth[b_ids, i_bottom_right, j_bottom_right]
    )

    kpts_depth[b_ids, k_ids] = interpolated_depth

    return kpts_depth
