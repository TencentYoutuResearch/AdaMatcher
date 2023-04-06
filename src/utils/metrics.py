import pdb
from collections import OrderedDict

import cv2
import numpy as np
import torch
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric
from loguru import logger

# import pydegensac

# --- METRICS ---


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.

    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ (E.T)  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (
        1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2)
        + 1.0 / (Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2)
    )  # N
    return d


def compute_symmetrical_epipolar_errors2(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    overlap_mask1_l1 = data["overlap_mask1_l1"]
    overlap_mask0_l1 = data["overlap_mask0_l1"]
    mask1_d8_size = data["mask1_d8"][data["mask1_d8"]].size()
    mask0_d8_size = data["mask0_d8"][data["mask0_d8"]].size()

    overlap_scores1 = overlap_mask1_l1.sum() / (mask1_d8_size[0] * mask1_d8_size[1])
    overlap_scores0 = overlap_mask0_l1.sum() / (mask0_d8_size[0] * mask0_d8_size[1])

    times = 1.2
    if overlap_scores1 > overlap_scores0 * times:
        m_bids = data["b_ids1_l2"]
        pts1 = data["kpts1_l2"]
        pts0 = data["kpts0from1_l2"]
        scores = data["std0"]
    elif overlap_scores0 > overlap_scores1 * times:
        m_bids = data["b_ids0_l2"]
        pts1 = data["kpts1from0_l2"]
        pts0 = data["kpts0_l2"]
        scores = data["std1"]
    else:
        m_bids = torch.cat([data["b_ids1_l2"], data["b_ids0_l2"]], dim=0)
        pts1 = torch.cat([data["kpts1_l2"], data["kpts1from0_l2"]], dim=0)
        pts0 = torch.cat([data["kpts0from1_l2"], data["kpts0_l2"]], dim=0)
        if len(m_bids) > 512 * 4:
            scores = torch.cat([data["std0"], data["std1"]])
            scores, topk_index = torch.topk(scores, 512 * 4, largest=False)
            m_bids = m_bids[topk_index]
            pts1 = pts1[topk_index]
            pts0 = pts0[topk_index]

    # m_bids = data['m_bids']
    # pts0 = data['mkpts0_f']
    # pts1 = data['mkpts1_f']

    Tx = numeric.cross_product_matrix(data["T_0to1"][:, :3, 3])
    E_mat = Tx @ data["T_0to1"][:, :3, :3]

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(
                pts0[mask], pts1[mask], E_mat[bs], data["K0"][bs], data["K1"][bs]
            )
        )
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({"epi_errs": epi_errs})


def compute_symmetrical_epipolar_errors(data):
    """
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data["T_0to1"][:, :3, 3])
    E_mat = Tx @ data["T_0to1"][:, :3, :3]

    m_bids = data["m_bids"]
    pts0 = data["mkpts0_f"]
    pts1 = data["mkpts1_f"]

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(
                pts0[mask], pts1[mask], E_mat[bs], data["K0"][bs], data["K1"][bs]
            )
        )
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({"epi_errs": epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    # kpts0_ = (kpts0 - K0[[0, 1], [2, 2]][None])*K1[[0, 1], [0, 1]][None] / K0[[0, 1], [0, 1]][None]
    # kpts1_ = (kpts1 - K1[[0, 1], [2, 2]][None])
    # temp_K = np.array([[K1[0, 0], 0,    0],
    #                    [0,    K1[1, 1], 0],
    #                    [0,    0,        1]])

    # kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    # kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    # pdb.set_trace()
    transpose_flag = False
    if (
        np.linalg.norm(kpts0, axis=1, ord=2).mean()
        < np.linalg.norm(kpts1, axis=1, ord=2).mean()
        and 0
    ):
        transpose_flag = True
        kpts0_norm = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        kpts1_norm = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    else:
        kpts0_norm = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1_norm = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
        # kpts0_norm = np.concatenate([kpts0, np.ones_like(kpts0[:, [0]])], axis=-1)
        # kpts1_norm = np.concatenate([kpts1, np.ones_like(kpts1[:, [0]])], axis=-1)
        # kpts0_norm = (np.linalg.inv(K0) @ kpts0_norm.transpose(1, 0)).transpose(1, 0)[:, :2]
        # kpts1_norm = (np.linalg.inv(K1) @ kpts1_norm.transpose(1, 0)).transpose(1, 0)[:, :2]
    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    # ransac_thr = thresh / np.mean([np.max(K0[:2, :2]), np.max(K1[:2, :2])])
    # ransac_thr = thresh

    # pdb.set_trace()
    # compute pose with cv2
    # F_mat = torch.inverse(K1).transpose(1,2) @ E_mat @ torch.inverse(K0)
    # F2, m_ = pydegensac.findFundamentalMatrix(kpts0_, kpts1_, 0.3)
    # E2 = np.dot(np.dot(temp_K.T, F2), temp_K)  # K1.T @ F2 @ K0
    # temp_norm = np.linalg.norm(E2.reshape(-1,9), 2)
    # m_ = m_[:,None].astype('uint8')

    E, mask = cv2.findEssentialMat(
        kpts0_norm,
        kpts1_norm,
        np.eye(3),
        threshold=ransac_thr,
        prob=conf,
        method=cv2.RANSAC,
    )
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    # pdb.set_trace()
    for _E in np.split(E, len(E) / 3):
        if transpose_flag and 0:
            n, R, t, _ = cv2.recoverPose(
                _E.T, kpts1_norm, kpts0_norm, np.eye(3), 1e9, mask=mask
            )
        else:
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0_norm, kpts1_norm, np.eye(3), 1e9, mask=mask
            )
        # n, R2, t2, _ = cv2.recoverPose(E2, kpts0_, kpts1_, temp_K, 1e9, mask=m_)
        if n > best_num_inliers:
            # ret = (R, t[:, 0], mask.ravel() > 0, R2, t2[:, 0], m_.ravel() > 0)
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def compute_pose_errors(data, config):
    """
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({"R_errs": [], "t_errs": [], "inliers": []})

    m_bids = data["m_bids"].cpu().numpy()
    pts0 = data["mkpts0_f"].cpu().numpy()
    pts1 = data["mkpts1_f"].cpu().numpy()
    K0 = data["K0"].cpu().numpy()
    K1 = data["K1"].cpu().numpy()
    T_0to1 = data["T_0to1"].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        ret = estimate_pose(
            pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf
        )

        if ret is None:
            data["R_errs"].append(np.inf)
            data["t_errs"].append(np.inf)
            data["inliers"].append(np.array([]).astype(np.bool))
        else:
            # R, t, inliers, R2, t2, inliers2 = ret
            R, t, inliers = ret
            t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
            # t_err2, R_err2 = relative_pose_error(T_0to1[bs], R2, t2, ignore_gt_t_thr=0.0)
            # pdb.set_trace()
            data["R_errs"].append(R_err)
            data["t_errs"].append(t_err)
            data["inliers"].append(inliers)


def compute_coarse_error(data):
    overlap_scores1 = data["overlap_scores1"]
    overlap_scores0 = data["overlap_scores0"]
    matrix1 = data["mask1_scores"]  # [N, h0_l1*w0_l1, h1_l1*w1_l1]
    matrix0 = data["mask0_scores"]
    gt_matrix1 = data["spv_conf_matrix1_l1"]  # [N, h0_l1*w0_l1, h1_l1*w1_l1]
    gt_matrix0 = data["spv_conf_matrix0_l1"]

    N = len(matrix1)
    # pdb.set_trace()
    fp_scores = []
    miss_scores = []
    if "mask0_d8" in data:
        m_mask1 = (
            data["mask0_d8"].flatten(-2)[..., None]
            * data["mask1_d8"].flatten(-2)[:, None]
        ).float()
        m_mask0 = (
            data["mask1_d8"].flatten(-2)[..., None]
            * data["mask0_d8"].flatten(-2)[:, None]
        ).float()
        for bs_id in range(N):
            if overlap_scores1[bs_id] > overlap_scores0[bs_id]:
                gt_m = gt_matrix1[bs_id].bool()
                _m = matrix1[bs_id] > 0.5
                bs_fp = (
                    ((~gt_m) * _m * m_mask1[bs_id]).bool().sum().float()
                )  # /m_mask1[bs_id].sum()
                bs_tn = ((~gt_m) * (~_m) * m_mask1[bs_id]).bool().sum().float()
                bs_fn = (gt_m * (~_m) * m_mask1[bs_id]).bool().sum().float()
                bs_tp = (gt_m * _m * m_mask1[bs_id]).bool().sum().float()
            else:
                gt_m = gt_matrix0[bs_id].bool()
                _m = matrix0[bs_id] > 0.5
                bs_fp = (
                    ((~gt_m) * _m * m_mask0[bs_id]).bool().sum().float()
                )  # /m_mask0[bs_id].sum()
                bs_tn = ((~gt_m) * (~_m) * m_mask0[bs_id]).bool().sum().float()
                bs_fn = (gt_m * (~_m) * m_mask0[bs_id]).bool().sum().float()
                bs_tp = (gt_m * _m * m_mask0[bs_id]).bool().sum().float()
            bs_fp_scores = bs_fp / (bs_fp + bs_tn)
            bs_miss_scores = bs_fn / (bs_tp + bs_fn)
            fp_scores.append(bs_fp_scores.item())
            miss_scores.append(bs_miss_scores.item())
    else:
        for bs_id in range(N):
            if overlap_scores1[bs_id] > overlap_scores0[bs_id]:
                gt_m = gt_matrix1[bs_id].bool()
                _m = matrix1[bs_id] > 0.5
                bs_fp = ((~gt_m) * _m).bool().sum().float()  # /m_mask1[bs_id].sum()
                bs_tn = ((~gt_m) * (~_m)).bool().sum().float()
                bs_fn = (gt_m * (~_m)).bool().sum().float()
                bs_tp = (gt_m * _m).bool().sum().float()
            else:
                gt_m = gt_matrix0[bs_id].bool()
                _m = matrix0[bs_id] > 0.5
                bs_fp = ((~gt_m) * _m).bool().sum().float()  # /m_mask0[bs_id].sum()
                bs_tn = ((~gt_m) * (~_m)).bool().sum().float()
                bs_fn = (gt_m * (~_m)).bool().sum().float()
                bs_tp = (gt_m * _m).bool().sum().float()
            bs_fp_scores = bs_fp / (bs_fp + bs_tn)
            bs_miss_scores = bs_fn / (bs_tp + bs_fn)
            fp_scores.append(bs_fp_scores.item())
            miss_scores.append(bs_miss_scores.item())
    data.update({"fp_scores": fp_scores, "miss_scores": miss_scores})


# --- METRIC AGGREGATION ---
def error_fp_miss_scores(fp, miss):
    """
    Args:
        fp (list): [N,]
    """
    fp = [0] + sorted(list(fp))
    fp_index = list(np.linspace(0, 1, len(fp)))
    fp_rates = []
    fp_thresholds = [1e-7, 1e-6, 5e-6, 1e-5, 5e-5]
    for thr in fp_thresholds:
        last_index = np.searchsorted(fp, thr)
        y = fp_index[:last_index] + [fp_index[last_index - 1]]
        x = fp[:last_index] + [thr]
        fp_rates.append(np.trapz(y, x) / thr)

    miss = [0] + sorted(list(miss))
    miss_index = list(np.linspace(0, 1, len(miss)))
    miss_rates = []
    miss_thresholds = [0.25, 0.4, 0.5, 0.75, 0.95]
    for thr in miss_thresholds:
        last_index = np.searchsorted(miss, thr)
        y = miss[:last_index] + [miss_index[last_index - 1]]
        x = miss[:last_index] + [thr]
        miss_rates.append(np.trapz(y, x) / thr)

    return {
        **{f"fp@{t}": auc for t, auc in zip(fp_thresholds, fp_rates)},
        **{f"miss@{t}": auc for t, auc in zip(miss_thresholds, miss_rates)},
        **{
            "mean_fp": np.mean(fp[1:]),
            "min_fp": np.min(fp[1:]),
            "max_fp": np.max(fp[1:]),
        },
        **{
            "mean_miss": np.mean(miss[1:]),
            "min_miss": np.min(miss[1:]),
            "max_miss": np.max(miss[1:]),
        },
    }


def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f"prec@{t:.0e}": prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """Aggregate metrics for the whole dataset:

    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics["identifiers"]))
    unq_ids = list(unq_ids.values())
    logger.info(f"Aggregating metrics over {len(unq_ids)} unique items...")

    # fp, miss_rate
    # fp = np.array(metrics['fp_scores'], dtype=object)[unq_ids]
    # miss = np.array(metrics['miss_scores'], dtype=object)[unq_ids]
    # fp_miss_rates = error_fp_miss_scores(fp, miss)

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics["R_errs"], metrics["t_errs"]]), axis=0)[
        unq_ids
    ]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(
        np.array(metrics["epi_errs"], dtype=object)[unq_ids], dist_thresholds, True
    )  # (prec@err_thr)

    return {**aucs, **precs}
    # return {**aucs, **precs, **fp_miss_rates}
