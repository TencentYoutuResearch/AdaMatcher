import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseModule(nn.Module):
    def __init__(self, conf, resolution):
        super().__init__()

        self.scale_l0, self.scale_l1, self.scale_l2 = resolution
        self.patch_size = self.scale_l0 // self.scale_l1

        self.conf_threshold = conf["conf_threshold"]  # 0.5
        self.inference_conf_threshold = conf[
            "inference_conf_threshold"
        ]  # 0.5  # 0.55 # 0.6(megadepth)
        self.class_threshold = conf["class_threshold"]  # 0.2  # 0.3
        self.class_num_threshold = conf["class_num_threshold"]  # 0.2

        self.train_coarse_percent = 0.3
        self.train_pad_num_gt_min = 200
        self.max_train_pts = conf[
            "max_train_pts"
        ]  # 2500  # 2200(megadepth0306)  # 2500(best)  # 1500
        self.max_o_scale = conf["max_o_scale"]  # 5.   # 832:2.5 # 3. # 4.
        self.t_k = conf["t_k"]  # -1
        self.patch_limit_n = conf["patch_limit_n"]  # 5 # 15
        self.use_dual_filter = conf["use_dual_filter"]  # False

    def generate_matching_matrix(
        self, mask_feats, query_feats_all, m_mask, q_mask, path_size
    ):

        norm_size = path_size**2
        N, c_m, hm_l1, wm_l1 = mask_feats.size()
        _, c_q, hq_l1, wq_l1 = query_feats_all.size()

        if m_mask is not None:
            m_mask = m_mask.reshape(N, -1)
            q_mask = q_mask.reshape(N, -1)
        mask_feats = rearrange(mask_feats, "n c h w -> n (h w) c")  # [n, s, c]
        query_feats_all = rearrange(
            query_feats_all, "n c h w -> n (h w) c"
        )  # [n, q, c]
        mask_feats = mask_feats / c_m**0.5
        query_feats_all = query_feats_all / c_q**0.5

        sim_matrix = torch.einsum("nqc,nsc->nqs", query_feats_all, mask_feats) * 10
        # sim_matrix = mid_matrix * 10
        # sim_matrix.masked_fill_(
        #             ~(q_mask[..., None] * m_mask[:, None]).bool(),
        #             -INF)
        instance_masks = sim_matrix.softmax(dim=1)  # * sim_matrix.softmax(dim=2)
        query_masks = sim_matrix.softmax(dim=2).transpose(1, 2)

        with torch.no_grad():
            tmp_instance_masks_d = rearrange(
                instance_masks,
                "n (h ph w pw) s -> n (h w) (ph pw) s",
                h=hq_l1 // path_size,
                w=wq_l1 // path_size,
                ph=path_size,
                pw=path_size,
            )
            instance_masks_d = tmp_instance_masks_d.sum(dim=2).clamp(
                max=1 - 1e-6
            )  # / norm_size

            tmp_query_masks_d = rearrange(
                query_masks,
                "n (h ph w pw) s -> n (h w) (ph pw) s",
                h=hm_l1 // path_size,
                w=wm_l1 // path_size,
                ph=path_size,
                pw=path_size,
            )
            query_masks_d = tmp_query_masks_d.sum(dim=2).clamp(
                max=1 - 1e-6
            )  # / norm_size

        return (
            instance_masks,
            query_masks,
            instance_masks_d,
            query_masks_d,
            tmp_instance_masks_d,
            tmp_query_masks_d,
        )

    @torch.no_grad()
    def compute_overlap_scores(
        self, instance_scors1, class_b0_l0_ids, class_k0_l0_ids, uncalculate_mask1=None
    ):
        if len(class_k0_l0_ids) != 0:
            instance_mask1 = instance_scors1 > self.conf_threshold
            overlap_scores1 = []
            for bs_id in range(self.bs):
                bs_mask0_l0 = class_b0_l0_ids == bs_id
                if bs_mask0_l0.any():
                    overlap_m1_l1 = (
                        instance_mask1[bs_mask0_l0].sum(dim=0) > 0
                    )  # 从0_l0到1_l1
                    overlap_patch_num1 = torch.nonzero(
                        instance_mask1[bs_mask0_l0].sum(dim=-1), as_tuple=False
                    ).shape[0]
                    overlap_patch_num1 = max(overlap_patch_num1, 1)
                    overlap_s1 = overlap_m1_l1.sum().float() / torch.tensor(
                        overlap_patch_num1, device=self.device
                    )
                    overlap_scores1.append(overlap_s1)
                else:
                    overlap_scores1.append(torch.tensor(0.0, device=self.device))
            overlap_scores1 = torch.stack(overlap_scores1, dim=0)
        else:
            overlap_scores1 = torch.zeros((self.bs,), device=self.device)
        return overlap_scores1

    @torch.no_grad()
    def adaptive_matching_proposal(
        self,
        instance_mask1,
        gt1,
        mid_matrix1_l0l1,
        class_k0_l0_ids,
        hw1_d64,
        hw0_d64,
        hw1_d8,
        hw0_d8,
        mask1_d8=None,
        mask0_d8=None,
    ):
        """
        Args:
            instance_mask1: [k1, h1_l1*w1_l1] k1 from h0_l0*w0_l0
            mid_matrix1_l0l1: [k1, 4, h1_l1*w1_l1]
            class_k0_l0_ids: [k1, ]
        """
        patch_size_l0l1 = self.scale_l0 // self.scale_l1  # 8
        if self.training:  # 1:
            k_ids1_l1, j_ids1_l1 = torch.nonzero(
                instance_mask1 > self.conf_threshold, as_tuple=True
            )
            i_ids1_l0 = class_k0_l0_ids[k_ids1_l1]  # [k1', ]
        elif self.training and 0:
            limit_n = 40  # 15(best)  # 5  # 15
            sub_patch = mid_matrix1_l0l1.shape[1]
            values, inds_j = mid_matrix1_l0l1.topk(limit_n, dim=-1)
            inds_k = (
                torch.arange(class_k0_l0_ids.size(0))[:, None, None]
                .repeat(1, sub_patch, limit_n)
                .to(inds_j)
            )
            j_ids1_l1 = inds_j[
                values > 0.25
            ]  # [values > self.inference_conf_threshold]
            k_ids1_l1 = inds_k[
                values > 0.25
            ]  # [values > self.inference_conf_threshold]
            i_ids1_l0 = class_k0_l0_ids[k_ids1_l1]
        else:
            # limit_n = 15  # 15  # 20  # 15(best)  # 5  # 15
            sub_patch = mid_matrix1_l0l1.shape[1]
            values, inds_j = mid_matrix1_l0l1.topk(self.patch_limit_n, dim=-1)
            inds_k = (
                torch.arange(class_k0_l0_ids.size(0))[:, None, None]
                .repeat(1, sub_patch, self.patch_limit_n)
                .to(inds_j)
            )
            j_ids1_l1 = inds_j[values > self.inference_conf_threshold]
            k_ids1_l1 = inds_k[values > self.inference_conf_threshold]
            i_ids1_l0 = class_k0_l0_ids[k_ids1_l1]

        if len(j_ids1_l1) != 0:
            scores, i_ids1_l0l1 = mid_matrix1_l0l1[k_ids1_l1, :, j_ids1_l1].max(
                dim=-1
            )  # [k,64]->[k,1] index
            if not self.training:
                scores_mask = scores > self.inference_conf_threshold
                i_ids1_l0 = i_ids1_l0[scores_mask]
                i_ids1_l0l1 = i_ids1_l0l1[scores_mask]
                j_ids1_l1 = j_ids1_l1[scores_mask]
                scores = scores[scores_mask]
                if self.t_k > 0 and len(scores) > self.t_k:
                    topk_index = scores.topk(self.t_k)[1]
                    i_ids1_l0 = i_ids1_l0[topk_index]
                    i_ids1_l0l1 = i_ids1_l0l1[topk_index]
                    j_ids1_l1 = j_ids1_l1[topk_index]
                    scores = scores[topk_index]
        else:
            scores = i_ids1_l0l1 = j_ids1_l1

        ######## sample ########
        if gt1 is not None and self.training:
            gt_i_ids1_l0, gt_i_ids1_l0l1, gt_j_ids1_l1 = gt1
            if mask0_d8 is None:
                num_candidates_max = self.bs * max(
                    hw0_d8[0] * hw0_d8[1], hw1_d8[0] * hw1_d8[1]
                )
            else:
                num_candidates_max = compute_max_candidates(mask0_d8, mask1_d8)

            num_matches_train = min(
                int(num_candidates_max * self.train_coarse_percent), self.max_train_pts
            )  # 2500  # 1440 1500 1000  # int(num_candidates_max * self.train_coarse_percent)
            # num_matches_train = int(num_candidates_max * self.train_coarse_percent) # origin
            num_matches_pred = len(j_ids1_l1)
            assert (
                self.train_pad_num_gt_min < num_matches_train
            ), "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=self.device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min,),
                    device=self.device,
                )

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            if len(gt_j_ids1_l1) > 0:
                # if len(pred_indices) > 0:
                gt_pad_indices = torch.randint(
                    len(gt_j_ids1_l1),
                    (
                        max(
                            num_matches_train - num_matches_pred,
                            self.train_pad_num_gt_min,
                        ),
                    ),
                    device=self.device,
                )
                # else:
                #     gt_pad_indices = torch.randint(
                #         len(gt_j_ids1_l1),
                #         (num_matches_train, ),
                #         device=self.device)
            else:
                gt_pad_indices = torch.empty(0, device=self.device, dtype=torch.long)
            mconf_gt = torch.zeros(
                len(gt_i_ids1_l0), device=self.device
            )  # set conf of gt paddings to all zero

            i_ids1_l0, i_ids1_l0l1, j_ids1_l1, scores1 = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [i_ids1_l0, gt_i_ids1_l0],
                    [i_ids1_l0l1, gt_i_ids1_l0l1],
                    [j_ids1_l1, gt_j_ids1_l1],
                    [scores, mconf_gt],
                )
            )

        if len(j_ids1_l1) == 0:  # no points
            # assert self.training is False
            return (
                torch.empty(0, 2, device=instance_mask1.device),
                torch.empty(0, 2, device=instance_mask1.device),
                torch.empty(0, device=instance_mask1.device, dtype=torch.long),
                torch.empty(0, device=instance_mask1.device, dtype=torch.long),
                torch.empty(0, device=instance_mask1.device, dtype=torch.long),
                torch.empty(0, device=instance_mask1.device, dtype=torch.long),
            )

        kpts1_l1 = torch.stack([j_ids1_l1 % hw1_d8[1], j_ids1_l1 // hw1_d8[1]], dim=1)
        kpts0from1_l1 = torch.stack(
            [i_ids1_l0 % hw0_d64[1], i_ids1_l0 // hw0_d64[1]], dim=1
        ) * patch_size_l0l1 + torch.stack(
            [i_ids1_l0l1 % patch_size_l0l1, i_ids1_l0l1 // patch_size_l0l1], dim=1
        )
        i_ids1_l1 = kpts0from1_l1[:, 0] + kpts0from1_l1[:, 1] * hw0_d8[1]

        return kpts1_l1, kpts0from1_l1, i_ids1_l1, j_ids1_l1, i_ids1_l0, scores

    @torch.no_grad()
    def get_filtered_index(self, data):
        return_dict = {}
        cas_score0 = data["cas_score0"]
        pred_class0_l1 = cas_score0 > self.class_threshold
        pred_class0_l0 = rearrange(
            pred_class0_l1,
            "n (h ph) (w pw) -> n h w (ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )  # [bs, h0_l0, w0_l0, 64]
        if "mask0_l0" in data:
            pred_class0_l0 = (
                (pred_class0_l0.float().sum(dim=-1) * data["mask0_l0"])
                / self.patch_size**2
            ) > self.class_num_threshold  # [bs, h0_l0, w0_l0]
        else:
            pred_class0_l0 = (
                pred_class0_l0.float().sum(dim=-1) / self.patch_size**2
            ) > self.class_num_threshold

        cas_score1 = data["cas_score1"]
        pred_class1_l1 = cas_score1 > self.class_threshold
        pred_class1_l0 = rearrange(
            pred_class1_l1,
            "n (h ph) (w pw) -> n h w (ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        if "mask1_l0" in data:
            pred_class1_l0 = (
                (pred_class1_l0.float().sum(dim=-1) * data["mask1_l0"])
                / self.patch_size**2
            ) > self.class_num_threshold  # [bs, h1_l0, w1_l0]
        else:
            pred_class1_l0 = (
                pred_class1_l0.float().sum(dim=-1) / self.patch_size**2
            ) > self.class_num_threshold

        return_dict.update(
            {
                "pred_class0_l0": pred_class0_l0,
                "pred_class1_l0": pred_class1_l0,
                "pred_class0_l1": pred_class0_l1,
                "pred_class1_l1": pred_class1_l1,
            }
        )
        pred_class0_l0 = rearrange(pred_class0_l0, "n h w -> n (h w)")
        pred_class1_l0 = rearrange(pred_class1_l0, "n h w -> n (h w)")

        class_b0_l0_ids, class_k0_l0_ids = torch.nonzero(pred_class0_l0, as_tuple=True)
        class_b1_l0_ids, class_k1_l0_ids = torch.nonzero(pred_class1_l0, as_tuple=True)

        return_dict.update(
            {
                "pred_class_b0_l0_ids": class_b0_l0_ids,
                "pred_class_k0_l0_ids": class_k0_l0_ids,
                "pred_class_b1_l0_ids": class_b1_l0_ids,
                "pred_class_k1_l0_ids": class_k1_l0_ids,
            }
        )

        if self.training:
            spv_class_b0_l0_ids = data["spv_class_b0_l0_ids"]
            spv_class_k0_l0_ids = data["spv_class_k0_l0_ids"]
            spv_class_b1_l0_ids = data["spv_class_b1_l0_ids"]
            spv_class_k1_l0_ids = data["spv_class_k1_l0_ids"]

            # merge gt and pred for training
            for bs_id in range(data["image0"].size(0)):
                spv_class_k0_bs = spv_class_k0_l0_ids[spv_class_b0_l0_ids == bs_id]
                class_k0_bs = class_k0_l0_ids[class_b0_l0_ids == bs_id]
                neq_k0_bs = spv_class_k0_bs[
                    torch.nonzero(
                        ~spv_class_k0_bs.unsqueeze(1).eq(class_k0_bs).any(1),
                        as_tuple=True,
                    )
                ]
                class_k0_l0_ids = torch.cat([class_k0_l0_ids, neq_k0_bs])
                class_b0_l0_ids = torch.cat(
                    [class_b0_l0_ids, torch.full_like(neq_k0_bs, bs_id)]
                )

                spv_class_k1_bs = spv_class_k1_l0_ids[spv_class_b1_l0_ids == bs_id]
                class_k1_bs = class_k1_l0_ids[class_b1_l0_ids == bs_id]
                neq_k1_bs = spv_class_k1_bs[
                    torch.nonzero(
                        ~spv_class_k1_bs.unsqueeze(1).eq(class_k1_bs).any(1),
                        as_tuple=True,
                    )
                ]
                class_k1_l0_ids = torch.cat([class_k1_l0_ids, neq_k1_bs])
                class_b1_l0_ids = torch.cat(
                    [class_b1_l0_ids, torch.full_like(neq_k1_bs, bs_id)]
                )

            return_dict.update(
                {
                    "train_class_b0_l0_ids": class_b0_l0_ids,
                    "train_class_k0_l0_ids": class_k0_l0_ids,
                    "train_class_b1_l0_ids": class_b1_l0_ids,
                    "train_class_k1_l0_ids": class_k1_l0_ids,
                }
            )
        return (
            return_dict,
            class_b0_l0_ids,
            class_k0_l0_ids,
            class_b1_l0_ids,
            class_k1_l0_ids,
        )

    def forward(self, data, mask_feat0, mask_feat1):
        self.device = mask_feat0.device
        hw0_l0, hw1_l0 = data["hw0_l0"], data["hw1_l0"]
        bs, _, h0_l1, w0_l1 = mask_feat0.size()
        _, _, h1_l1, w1_l1 = mask_feat1.size()
        self.bs = bs
        mask0_d8, mask1_d8 = data.get("mask0_d8", None), data.get("mask1_d8", None)

        ######## 1. Get Overlapped Patch Index ##########################################
        (
            index_dict,
            class_b0_l0_ids,
            class_k0_l0_ids,
            class_b1_l0_ids,
            class_k1_l0_ids,
        ) = self.get_filtered_index(data)
        data.update(**index_dict)

        ######## 2. Generate Matching Matrix ##########################################
        # conf_matrix1:   [bs, l, s]         conf_matrix0:   [bs, s, l]
        # conf_matrix1_d: [bs, l//64, s]     conf_matrix0_d: [bs, s//64, l]
        # tmp_matrix1_d: [bs, l//64, 64, s]  tmp_matrix0_d: [bs, s//64, 64, l]
        (
            conf_matrix1,
            conf_matrix0,
            conf_matrix1_d,
            conf_matrix0_d,
            tmp_matrix1_d,
            tmp_matrix0_d,
        ) = self.generate_matching_matrix(
            mask_feat1, mask_feat0, mask1_d8, mask0_d8, self.patch_size
        )  # [n, l, s]
        data.update(
            {
                "mask0_scores": conf_matrix0,  # [bs, h1_l1*w1_l1, h0_l1*w0_l1]
                "mask1_scores": conf_matrix1,  # [bs, h0_l1*w0_l1, h1_l1*w1_l1]
            }
        )

        if mask1_d8 is not None:
            uncalculate_mask1 = mask1_d8[class_b0_l0_ids].flatten(-2)
            instance_mask1 = (
                conf_matrix1_d[class_b0_l0_ids, class_k0_l0_ids] * uncalculate_mask1
            )
            mid_matrix1_l0l1 = tmp_matrix1_d[
                class_b0_l0_ids, class_k0_l0_ids
            ] * uncalculate_mask1.unsqueeze(1)
            data.update(
                {
                    "uncalculate_mask1": uncalculate_mask1,
                    "instance_mask1": instance_mask1,  # [k, h1_l1*w1_l1]
                    "mid_matrix1_l0l1": mid_matrix1_l0l1,
                }
            )
        else:
            instance_mask1 = conf_matrix1_d[class_b0_l0_ids, class_k0_l0_ids]
            mid_matrix1_l0l1 = tmp_matrix1_d[class_b0_l0_ids, class_k0_l0_ids]
            data.update(
                {
                    "instance_mask1": instance_mask1,  # [k, h1_l1*w1_l1]
                    "mid_matrix1_l0l1": mid_matrix1_l0l1,
                }
            )

        if mask0_d8 is not None:
            uncalculate_mask0 = mask0_d8[class_b1_l0_ids].flatten(-2)
            instance_mask0 = (
                conf_matrix0_d[class_b1_l0_ids, class_k1_l0_ids] * uncalculate_mask0
            )
            mid_matrix0_l0l1 = tmp_matrix0_d[
                class_b1_l0_ids, class_k1_l0_ids
            ] * uncalculate_mask0.unsqueeze(1)
            data.update(
                {
                    "uncalculate_mask0": uncalculate_mask0,
                    "instance_mask0": instance_mask0,  # [k, h0_l1*w0_l1]
                    "mid_matrix0_l0l1": mid_matrix0_l0l1,
                }
            )
        else:
            instance_mask0 = conf_matrix0_d[class_b1_l0_ids, class_k1_l0_ids]
            mid_matrix0_l0l1 = tmp_matrix0_d[class_b1_l0_ids, class_k1_l0_ids]
            data.update(
                {
                    "instance_mask0": instance_mask0,  # [k, h0_l1*w0_l1]
                    "mid_matrix0_l0l1": mid_matrix0_l0l1,
                }
            )

        ######## 3. Calculate Overlap Scores #####################################
        overlap_scores1 = self.compute_overlap_scores(
            instance_mask1, class_b0_l0_ids, class_k0_l0_ids
        )
        overlap_scores0 = self.compute_overlap_scores(
            instance_mask0, class_b1_l0_ids, class_k1_l0_ids
        )
        data.update(
            {"overlap_scores1": overlap_scores1, "overlap_scores0": overlap_scores0}
        )

        ######## 4. Adaptive Assignment ##########################################
        kpts1_l1, kpts0from1_l1 = [], []
        b_ids1_l1, i_ids1_l1, j_ids1_l1, i_ids1_l0 = [], [], [], []
        kpts0_l1, kpts1from0_l1 = [], []
        b_ids0_l1, i_ids0_l1, j_ids0_l1, i_ids0_l0 = [], [], [], []

        c_scores0, c_scores1, scales, scores = [], [], [], []
        m_bids, c_pts0, c_pts1 = [], [], []
        for bs_id in range(bs):
            bs_mask1_d8 = data["mask1_d8"][[bs_id]] if "mask1_d8" in data else None
            bs_mask0_d8 = data["mask0_d8"][[bs_id]] if "mask0_d8" in data else None
            if overlap_scores1[bs_id] > overlap_scores0[bs_id]:
                if overlap_scores0[bs_id] > 0:
                    o_scale0 = (overlap_scores1[bs_id] / overlap_scores0[bs_id]).item()
                    o_scale0 = max(min(o_scale0, self.max_o_scale), 1.0)
                else:
                    o_scale0 = self.max_o_scale
                scales.append(o_scale0)

                if self.training:
                    gt1 = (
                        data["spv_i_ids1_l0"][data["spv_b_ids1_l1"] == bs_id],
                        data["spv_i_ids1_l0l1"][data["spv_b_ids1_l1"] == bs_id],
                        data["spv_j_ids1_l1"][data["spv_b_ids1_l1"] == bs_id],
                    )
                else:
                    gt1 = None
                bs_mask0 = class_b0_l0_ids == bs_id
                bs_mid_matrix1_l0l1 = mid_matrix1_l0l1[bs_mask0]
                bs_class_k0_l0_ids = class_k0_l0_ids[bs_mask0]
                bs_instance_mask1 = instance_mask1[bs_mask0]  # > self.conf_threshold
                (
                    bs_kpts1_l1,
                    bs_kpts0from1_l1,
                    bs_i_ids1_l1,
                    bs_j_ids1_l1,
                    bs_i_ids1_l0,
                    scores1,
                ) = self.adaptive_matching_proposal(
                    bs_instance_mask1,
                    gt1,
                    bs_mid_matrix1_l0l1,
                    bs_class_k0_l0_ids,
                    hw1_l0,
                    hw0_l0,
                    (h1_l1, w1_l1),
                    (h0_l1, w0_l1),
                    mask1_d8=bs_mask1_d8,
                    mask0_d8=bs_mask0_d8,
                )
                bs_b_ids1_l1 = torch.full_like(bs_i_ids1_l1, bs_id)
                kpts1_l1.append(bs_kpts1_l1)
                kpts0from1_l1.append(bs_kpts0from1_l1)
                b_ids1_l1.append(bs_b_ids1_l1)
                i_ids1_l1.append(bs_i_ids1_l1)
                j_ids1_l1.append(bs_j_ids1_l1)
                i_ids1_l0.append(bs_i_ids1_l0)
                c_scores1.append(scores1)
            else:
                if overlap_scores1[bs_id] > 0:
                    o_scale1 = (overlap_scores0[bs_id] / overlap_scores1[bs_id]).item()
                    o_scale1 = max(min(o_scale1, self.max_o_scale), 1.0)
                else:
                    o_scale1 = self.max_o_scale
                scales.append(o_scale1)

                if self.training:
                    gt0 = (
                        data["spv_j_ids0_l0"][data["spv_b_ids0_l1"] == bs_id],
                        data["spv_j_ids0_l0l1"][data["spv_b_ids0_l1"] == bs_id],
                        data["spv_i_ids0_l1"][data["spv_b_ids0_l1"] == bs_id],
                    )
                else:
                    gt0 = None
                bs_mask1 = class_b1_l0_ids == bs_id
                bs_mid_matrix0_l0l1 = mid_matrix0_l0l1[bs_mask1]
                bs_class_k1_l0_ids = class_k1_l0_ids[bs_mask1]
                bs_instance_mask0 = instance_mask0[bs_mask1]  # > self.conf_threshold
                (
                    bs_kpts0_l1,
                    bs_kpts1from0_l1,
                    bs_i_ids0_l1,
                    bs_j_ids0_l1,
                    bs_i_ids0_l0,
                    scores0,
                ) = self.adaptive_matching_proposal(
                    bs_instance_mask0,
                    gt0,
                    bs_mid_matrix0_l0l1,
                    bs_class_k1_l0_ids,
                    hw0_l0,
                    hw1_l0,
                    (h0_l1, w0_l1),
                    (h1_l1, w1_l1),
                    mask1_d8=bs_mask0_d8,
                    mask0_d8=bs_mask1_d8,
                )
                bs_b_ids0_l1 = torch.full_like(bs_i_ids0_l1, bs_id)
                kpts0_l1.append(bs_kpts0_l1)
                kpts1from0_l1.append(bs_kpts1from0_l1)
                b_ids0_l1.append(bs_b_ids0_l1)
                i_ids0_l1.append(bs_i_ids0_l1)
                j_ids0_l1.append(bs_j_ids0_l1)
                i_ids0_l0.append(bs_i_ids0_l0)
                c_scores0.append(scores0)

        ######## 5. Update Data ##################################################
        data.update(
            dict(
                kpts1_l1=torch.cat(kpts1_l1)
                if b_ids1_l1
                else torch.empty(0, 2, device=self.device, dtype=torch.long),
                kpts0from1_l1=torch.cat(kpts0from1_l1)
                if b_ids1_l1
                else torch.empty(0, 2, device=self.device),
                b_ids1_l1=torch.cat(b_ids1_l1)
                if b_ids1_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                i_ids1_l1=torch.cat(i_ids1_l1)
                if b_ids1_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                j_ids1_l1=torch.cat(j_ids1_l1)
                if b_ids1_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                i_ids1_l0=torch.cat(i_ids1_l0)
                if b_ids1_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                c_scores1=torch.cat(c_scores1)
                if b_ids1_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
            )
        )

        data.update(
            dict(
                kpts0_l1=torch.cat(kpts0_l1)
                if b_ids0_l1
                else torch.empty(0, 2, device=self.device, dtype=torch.long),
                kpts1from0_l1=torch.cat(kpts1from0_l1)
                if b_ids0_l1
                else torch.empty(0, 2, device=self.device),
                b_ids0_l1=torch.cat(b_ids0_l1)
                if b_ids0_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                i_ids0_l1=torch.cat(i_ids0_l1)
                if b_ids0_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                j_ids0_l1=torch.cat(j_ids0_l1)
                if b_ids0_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                i_ids0_l0=torch.cat(i_ids0_l0)
                if b_ids0_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
                c_scores0=torch.cat(c_scores0)
                if b_ids0_l1
                else torch.empty(0, device=self.device, dtype=torch.long),
            )
        )

        if not self.training and self.use_dual_filter:
            self.dual_filter(data)

        if b_ids1_l1:
            m_bids.append(data["b_ids1_l1"])
            c_pts0.append(data["kpts0from1_l1"])
            c_pts1.append(data["kpts1_l1"])
            scores.append(data["c_scores1"])
        if b_ids0_l1:
            m_bids.append(data["b_ids0_l1"])
            c_pts1.append(data["kpts1from0_l1"])
            c_pts0.append(data["kpts0_l1"])
            scores.append(data["c_scores0"])

        if m_bids:
            scale0_l1 = (
                self.scale_l1 * data["scale0"][m_bids]
                if "scale0" in data
                else self.scale_l1
            )
            scale1_l1 = (
                self.scale_l1 * data["scale1"][m_bids]
                if "scale1" in data
                else self.scale_l1
            )
        else:
            scale0_l1 = scale1_l1 = 0.0
        # from IPython import embed

        # embed()
        data.update(
            dict(
                # scales=torch.cat(scales, dim=0),
                scales=scales,
                m_bids=torch.cat(m_bids)
                if m_bids
                else torch.empty(0, device=self.device, dtype=torch.long),
                mkpts0_c=torch.cat(c_pts0) * scale0_l1
                if m_bids
                else torch.empty(0, 2, device=self.device),
                mkpts1_c=torch.cat(c_pts1) * scale1_l1
                if m_bids
                else torch.empty(0, 2, device=self.device),
                scores=torch.cat(scores)
                if m_bids
                else torch.empty(0, device=self.device),
            )
        )

    @torch.no_grad()
    def dual_filter(self, data):
        pred_class0_l0 = data["pred_class0_l0"]
        pred_class1_l0 = data["pred_class1_l0"]
        mask0_l1 = rearrange(
            pred_class0_l0.unsqueeze(-1).repeat(1, 1, 1, self.patch_size**2),
            "n h w (ph pw) -> n (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        ).flatten(1)
        mask1_l1 = rearrange(
            pred_class1_l0.unsqueeze(-1).repeat(1, 1, 1, self.patch_size**2),
            "n h w (ph pw) -> n (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        ).flatten(1)
        if len(data["b_ids0_l1"]):
            b_ids0_l1 = data["b_ids0_l1"]
            i_ids0_l1 = data["i_ids0_l1"]
            j_ids0_l1 = data["j_ids0_l1"]
            dual_mask = mask0_l1[b_ids0_l1, j_ids0_l1] * mask1_l1[b_ids0_l1, i_ids0_l1]
            data["b_ids0_l1"] = b_ids0_l1[dual_mask]
            data["i_ids0_l1"] = i_ids0_l1[dual_mask]
            data["j_ids0_l1"] = j_ids0_l1[dual_mask]
            data["kpts0_l1"] = data["kpts0_l1"][dual_mask]
            data["kpts1from0_l1"] = data["kpts1from0_l1"][dual_mask]
            data["i_ids0_l0"] = data["i_ids0_l0"][dual_mask]
            data["c_scores0"] = data["c_scores0"][dual_mask]
        if len(data["b_ids1_l1"]):
            b_ids1_l1 = data["b_ids1_l1"]
            i_ids1_l1 = data["i_ids1_l1"]
            j_ids1_l1 = data["j_ids1_l1"]
            dual_mask = mask0_l1[b_ids1_l1, i_ids1_l1] * mask1_l1[b_ids1_l1, j_ids1_l1]
            data["b_ids1_l1"] = b_ids1_l1[dual_mask]
            data["i_ids1_l1"] = i_ids1_l1[dual_mask]
            data["j_ids1_l1"] = j_ids1_l1[dual_mask]
            data["kpts1_l1"] = data["kpts1_l1"][dual_mask]
            data["kpts0from1_l1"] = data["kpts0from1_l1"][dual_mask]
            data["i_ids1_l0"] = data["i_ids1_l0"][dual_mask]
            data["c_scores1"] = data["c_scores1"][dual_mask]
