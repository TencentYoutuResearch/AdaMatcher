import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fvcore.nn import sigmoid_focal_loss_jit
from loguru import logger

from src.adamatcher.utils.geometry import pose2fundamental
from src.utils.comm import reduce_mean

# from kornia.geometry.epipolar import numeric


class AdaMatcherLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.count = 0
        self.scale_l0, self.scale_l1, self.scale_l2 = config['adamatcher'][
            'resolution']  # 64, 8, 2
        self.scale_l0l1 = self.scale_l0 // self.scale_l1
        self.window_size = 5  # 5

        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.mask_weight = 1.0
        self.class_weight = 0.5
        self.fine_weight = 1.0
        self.use_neg = True  # False  # True(not finetune)
        self.fine_v_num = torch.tensor(0, device='cpu', dtype=torch.float)
        self.fine_pv_num = torch.tensor(0, device='cpu', dtype=torch.float)
        self.fine_all_num0 = torch.tensor(0, device='cpu', dtype=torch.float)
        self.fine_all_num1 = torch.tensor(0, device='cpu', dtype=torch.float)
        self.use_epipolarloss = False  # True  # False

    def dice_coefficient(self, x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x**2.0).sum(dim=1) + (target**2.0).sum(dim=1) + eps
        # loss = 1. - (2 * intersection / union)
        # return loss
        dice = 2 * intersection / union
        return dice

    def mask_focal_loss(self, x, target, mask=None, use_neg=True):
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
        alpha = 0.25
        gamma = 2.0
        # assert x.size(0)==target.size(0)==mask.size(0)
        pos_mask, neg_mask = target == 1, target == 0
        loss_pos = -alpha * torch.pow(1 - x[pos_mask],
                                      gamma) * (x[pos_mask]).log()
        # loss_neg = - alpha * torch.pow(x[neg_mask], gamma) * (1 - x[neg_mask]).log()
        loss_neg = (-(1 - alpha) * torch.pow(x[neg_mask], gamma) *
                    (1 - x[neg_mask]).log())

        if mask is not None:
            loss_pos = loss_pos * mask[pos_mask]
            loss_neg = loss_neg * mask[neg_mask]
        if use_neg:
            c_pos_w = c_neg_w = 1.0
        else:
            c_pos_w, c_neg_w = 1.0, 0.0
        # return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        if len(loss_pos) > 0 and len(loss_neg) > 0:
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif len(loss_pos) > 0:
            return c_pos_w * loss_pos.mean()
        else:
            return c_neg_w * loss_neg.mean()

    def class_focal_loss(self, x, target, mask=None):
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
        alpha = 0.25
        gamma = 2.0
        # pdb.set_trace()
        if mask is not None:
            pos_mask, neg_mask = (target == 1) * mask, (target == 0) * mask
        else:
            pos_mask, neg_mask = (target == 1), (target == 0)
        loss_pos = -alpha * torch.pow(1 - x[pos_mask],
                                      gamma) * (x[pos_mask]).log()
        # loss_neg = - alpha * torch.pow(x[neg_mask], gamma) * (1 - x[neg_mask]).log()
        loss_neg = (-(1 - alpha) * torch.pow(x[neg_mask], gamma) *
                    (1 - x[neg_mask]).log())

        loss_pos = loss_pos
        loss_neg = loss_neg
        c_pos_w = c_neg_w = 1.0
        # return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        if len(loss_pos) > 0 and len(loss_neg) > 0:
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif len(loss_pos) > 0:
            return c_pos_w * loss_pos.mean()
        else:
            return c_neg_w * loss_neg.mean()

    def focal_loss(self, x, target):
        # x, target:[batch, c, h_l0, w_l0]
        N, num_class, h_l0, w_l0 = x.size()
        num_pos_local = target[:, 1, :, :].sum()
        num_pos_avg = max(reduce_mean(num_pos_local).item(), 1.0)

        # Reshape: (N, c, h_l0, w_l0) -> (N, h_l0, w_l0, c) -> (N*h_l0*w_l0, c)
        pred = x.permute(0, 2, 3, 1).reshape(-1, num_class)
        target = target.permute(0, 2, 3, 1).reshape(-1, num_class)
        class_loss = sigmoid_focal_loss_jit(
            pred,
            target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction='sum',
        )
        return class_loss / num_pos_avg

    def _compute_fine_loss_l2(
        self,
        gt_r_w_pt1,
        pt1,
        w_pt1,
        r_w_pt1,
        b_ids1,
        std0,
        scale1,
        scale0,
        patch_size,
        F=None,
    ):

        weight = self.set_weight(std0)
        correct_mask = torch.norm(gt_r_w_pt1, p=float('inf'), dim=1) < 1.0
        self.fine_v_num += correct_mask.sum().float().cpu()
        if not correct_mask.any():
            if (
                    self.training
            ):  # this seldom happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                logger.warning(
                    'assign a false supervision to avoid ddp deadlock')
                # pdb.set_trace()
            loss_valid = ((r_w_pt1[0] - gt_r_w_pt1[0])**2).sum(-1) * 0.0
        else:
            loss_valid = (((r_w_pt1[correct_mask] - gt_r_w_pt1[correct_mask])**
                           2).sum(-1) * weight[correct_mask]).mean()

        if self.use_epipolarloss:
            loss_unvalid = []
            uncorrect_mask = (
                correct_mask  # torch.ones_like(correct_mask)  # ~correct_mask
            )
            for bs_id in range(self.bs):
                bs_mask = b_ids1[uncorrect_mask] == bs_id
                if bs_mask.any():
                    unvalid_pt1 = pt1[uncorrect_mask][bs_mask] * scale1[bs_id]
                    unvalid_w_pt1 = w_pt1[uncorrect_mask][bs_mask] * scale0[
                        bs_id]
                    unvalid_epipolar_cost = self.epipolar_cost(
                        unvalid_pt1.float(), unvalid_w_pt1, F[bs_id])
                    mask_unvalid = unvalid_epipolar_cost < patch_size
                    self.fine_pv_num += mask_unvalid.sum().float().cpu()
                    if mask_unvalid.any():
                        unvalid_weight = weight[uncorrect_mask][bs_mask][
                            mask_unvalid]
                        loss_unvalid.append(
                            torch.mean(unvalid_epipolar_cost[mask_unvalid] *
                                       unvalid_weight) / (patch_size))

            # if loss_valid is not None:
            if len(loss_unvalid) != 0:
                return loss_valid + torch.stack(loss_unvalid, dim=0).mean()
            else:
                return loss_valid
            # else:
            #     if len(loss_unvalid) != 0:
            #         return torch.stack(loss_unvalid, dim=0).mean()
            #     else:
            #         return ((r_w_pt1[0]-gt_r_w_pt1[0])** 2).sum(-1) * 0.
        else:
            # if loss_valid is not None:
            return loss_valid
            # else:
            #     return ((r_w_pt1[0]-gt_r_w_pt1[0])** 2).sum(-1) * 0.

    def set_weight(self, std, mask=None, regularizer=0.0):
        inverse_std = 1.0 / torch.clamp(std + regularizer, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # [n]

        # if mask is not None:
        #     weight *= mask.float()
        #     weight /= (torch.mean(weight) + 1e-8)
        return weight

    def homogenize(self, coord):
        coord = torch.cat((coord, torch.ones_like(coord[:, [0]])), -1)
        return coord

    def epipolar_cost(self, coord1, coord2, fmatrix):
        coord1_h = self.homogenize(coord1).transpose(0, 1)  # [3, k]
        coord2_h = self.homogenize(coord2).transpose(0, 1)  # [3, k]
        epipolar_line = fmatrix @ coord1_h  # .bmm(coord1_h)  # [3, k]
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:2, :], dim=0, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_,
                                             dim=0))  # [k]
        return essential_cost

    def mid_spv(self, mid_feat0, mid_feat1, gt_matrix, weights):
        c = mid_feat0.shape[-1]
        matrix = (torch.einsum('nqc,nsc->nqs', mid_feat0 / c**0.5,
                               mid_feat1 / c**0.5) * 10).softmax(dim=1)
        return self.mask_focal_loss(matrix, gt_matrix, weights)

    def forward(self, data):
        # pdb.set_trace()
        self.bs = data['image0'].size(0)
        self.fine_v_num = torch.tensor(0, device='cpu', dtype=torch.float)
        self.fine_pv_num = torch.tensor(0, device='cpu', dtype=torch.float)
        if 'scale1' in data:
            s1_l2 = self.scale_l2 * data['scale1']  # [:, None]
            s0_l2 = self.scale_l2 * data['scale0']  # [:, None]
        else:
            s1_l2 = torch.tensor(
                [[self.scale_l2, self.scale_l2]],
                dtype=torch.float,
                device=data['K0'].device,
            ).repeat(self.bs, 1)
            s0_l2 = torch.tensor(
                [[self.scale_l2, self.scale_l2]],
                dtype=torch.float,
                device=data['K0'].device,
            ).repeat(self.bs, 1)

        # co-visible area segmentation loss
        cas_score0 = data['cas_score0']  # [batch, 1, h0_l1, w0_l1]
        cas_score1 = data['cas_score1']  # [batch, 1, h1_l1, w1_l1]
        gt_cas0 = data['spv_class_l1_gt0']  # [batch, 2, h0_l1, w0_l1]
        gt_cas1 = data['spv_class_l1_gt1']  # [batch, 2, h1_l1, w1_l1]
        cas_loss0 = self.class_focal_loss(cas_score0, gt_cas0,
                                          data.get('mask0_d8', None))
        cas_loss1 = self.class_focal_loss(cas_score1, gt_cas1,
                                          data.get('mask1_d8', None))
        cas_loss = cas_loss0 + cas_loss1

        if 'train_class_b0_l0_ids' in data:
            class_b1_l0_ids = data['train_class_b1_l0_ids']
            class_k1_l0_ids = data['train_class_k1_l0_ids']
            class_b0_l0_ids = data['train_class_b0_l0_ids']
            class_k0_l0_ids = data['train_class_k0_l0_ids']
        else:
            class_b1_l0_ids = data['pred_class_b1_l0_ids']
            class_k1_l0_ids = data['pred_class_k1_l0_ids']
            class_b0_l0_ids = data['pred_class_b0_l0_ids']
            class_k0_l0_ids = data['pred_class_k0_l0_ids']

        # coarse level
        if len(class_k0_l0_ids) != 0 and len(class_k1_l0_ids) != 0:
            # mask loss
            mask0_scores = data['mask0_scores']  # [N, s, l]
            mask1_scores = data['mask1_scores']  # [N, l, s]
            if 'mask1_d8' in data:
                weight0 = (data['mask1_d8'].flatten(-2)[..., None] *
                           data['mask0_d8'].flatten(-2)[:, None]).float()
                weight1 = (data['mask0_d8'].flatten(-2)[..., None] *
                           data['mask1_d8'].flatten(-2)[:, None]).float()

            if 1:
                mask_dice_loss, mask_focal_loss = [], []
                # overlap_scores1, overlap_scores0 = data['overlap_scores1'], data['overlap_scores0']
                for bs_id in range(self.bs):
                    spv_b_ids1_l1 = data['spv_b_ids1_l1']
                    spv_b_ids0_l1 = data['spv_b_ids0_l1']
                    if len(spv_b_ids1_l1[spv_b_ids1_l1 == bs_id]) > len(
                            spv_b_ids0_l1[spv_b_ids0_l1 == bs_id]
                    ):  # overlap_scores1[bs_id] > overlap_scores0[bs_id]:
                        if 'mask1_d8' in data:
                            bs_mask_focal_loss = self.mask_focal_loss(
                                mask1_scores[[bs_id]],
                                data['spv_conf_matrix1_l1'][[bs_id]],
                                weight1[[bs_id]],
                                use_neg=self.use_neg,
                            )  # \
                        else:
                            bs_mask_focal_loss = self.mask_focal_loss(
                                mask1_scores[[bs_id]],
                                data['spv_conf_matrix1_l1'][[bs_id]],
                                use_neg=self.use_neg,
                            )
                    else:
                        if 'mask0_d8' in data:
                            bs_mask_focal_loss = self.mask_focal_loss(
                                mask0_scores[[bs_id]],
                                data['spv_conf_matrix0_l1'][[bs_id]],
                                weight0[[bs_id]],
                                use_neg=self.use_neg,
                            )  # \
                        else:
                            bs_mask_focal_loss = self.mask_focal_loss(
                                mask0_scores[[bs_id]],
                                data['spv_conf_matrix0_l1'][[bs_id]],
                                use_neg=self.use_neg,
                            )
                    mask_focal_loss.append(bs_mask_focal_loss)
                mask_focal_loss = torch.stack(mask_focal_loss, dim=0).mean()
                mask_loss = mask_focal_loss

            coarse_loss = self.class_weight * cas_loss + self.mask_weight * mask_loss
        else:
            coarse_loss = None

        # fine level
        spv_w_pt0_i_l2, spv_pt0_i_l2 = (
            data['spv_w_pt0_i_l1'] / s1_l2.unsqueeze(1),
            (data['spv_pt0_i_l1'] / s0_l2.unsqueeze(1)).round(),
        )
        spv_w_pt1_i_l2, spv_pt1_i_l2 = (
            data['spv_w_pt1_i_l1'] / s0_l2.unsqueeze(1),
            (data['spv_pt1_i_l1'] / s1_l2.unsqueeze(1)).round(),
        )

        b_ids0_l1, i_ids0_l1, j_ids0_l1 = (
            data['b_ids0_l2'],
            data['i_ids0_l1'],
            data['j_ids0_l1'],
        )
        if len(b_ids0_l1) > 0:
            gt_pt0_l2 = spv_pt0_i_l2[b_ids0_l1, j_ids0_l1]
            pt0 = data['kpts0_l2']  # * s0_l2
            p_mask0 = (pt0 == gt_pt0_l2).all(-1)
            pt0, gt_pt0_l2, b_ids0_l1, i_ids0_l1, j_ids0_l1 = (
                pt0[p_mask0],
                gt_pt0_l2[p_mask0],
                b_ids0_l1[p_mask0],
                i_ids0_l1[p_mask0],
                j_ids0_l1[p_mask0],
            )

            std1 = data['std1'][p_mask0]
            w_pt0 = data['kpts1from0_l2'][p_mask0]  # * s1_l2
            r_w_pt0 = data['relative_kpts1from0_l2'][p_mask0]
            patch1_center_coord = data['patch1_center_coord_l2'][p_mask0]
            gt_w_pt0_l2 = spv_w_pt0_i_l2[b_ids0_l1, j_ids0_l1]
            gt_r_w_pt0_l2 = (gt_w_pt0_l2 -
                             patch1_center_coord) / (self.window_size // 2)

            F_0to1 = pose2fundamental(data['K0'], data['K1'], data['T_0to1'])
            fine_loss0 = self._compute_fine_loss_l2(
                gt_r_w_pt0_l2,
                pt0,
                w_pt0,
                r_w_pt0,
                b_ids0_l1,
                std1,
                s0_l2,
                s1_l2,
                self.window_size * 0.5 * torch.norm(s1_l2, p=2),
                F_0to1,
            )  # /2  0.8
            if fine_loss0 is None:
                fine_loss0 = torch.zeros_like(cas_loss)
        else:
            fine_loss0 = torch.zeros_like(cas_loss)

        b_ids1_l1, i_ids1_l1, j_ids1_l1 = (
            data['b_ids1_l2'],
            data['i_ids1_l1'],
            data['j_ids1_l1'],
        )
        if len(b_ids1_l1) > 0:
            pt1 = data['kpts1_l2']  # * s1_l2
            gt_pt1_l2 = spv_pt1_i_l2[b_ids1_l1, j_ids1_l1]
            p_mask1 = (pt1 == gt_pt1_l2).all(-1)
            pt1, gt_pt1_l2, b_ids1_l1, i_ids1_l1, j_ids1_l1 = (
                pt1[p_mask1],
                gt_pt1_l2[p_mask1],
                b_ids1_l1[p_mask1],
                i_ids1_l1[p_mask1],
                j_ids1_l1[p_mask1],
            )

            w_pt1 = data['kpts0from1_l2'][p_mask1]  # * s0_l2
            r_w_pt1 = data['relative_kpts0from1_l2'][p_mask1]
            patch0_center_coord = data['patch0_center_coord_l2'][p_mask1]
            std0 = data['std0'][p_mask1]
            gt_w_pt1_l2 = spv_w_pt1_i_l2[b_ids1_l1, j_ids1_l1]
            gt_r_w_pt1_l2 = (gt_w_pt1_l2 -
                             patch0_center_coord) / (self.window_size // 2)

            F_1to0 = pose2fundamental(data['K1'], data['K0'], data['T_1to0'])
            fine_loss1 = self._compute_fine_loss_l2(
                gt_r_w_pt1_l2,
                pt1,
                w_pt1,
                r_w_pt1,
                b_ids1_l1,
                std0,
                s1_l2,
                s0_l2,
                self.window_size * 0.5 * torch.norm(s0_l2, p=2),
                F_1to0,
            )  # /2 0.8
            if fine_loss1 is None:
                fine_loss1 = torch.zeros_like(cas_loss)
        else:
            fine_loss1 = torch.zeros_like(cas_loss)

        fine_loss = (fine_loss0 + fine_loss1) * self.fine_weight

        self.fine_all_num0 = torch.tensor(len(b_ids0_l1),
                                          device='cpu',
                                          dtype=torch.float)
        self.fine_all_num1 = torch.tensor(len(b_ids1_l1),
                                          device='cpu',
                                          dtype=torch.float)

        if coarse_loss is not None:
            loss = coarse_loss + fine_loss  # 1.5*fine_loss
            loss_scalars = {
                'mf':
                mask_focal_loss.clone().detach().cpu(
                ),  # (mask_focal_loss0.clone().detach().cpu() + mask_focal_loss1.clone().detach().cpu()) / 2,
                'c': (cas_loss0.clone().detach().cpu() +
                      cas_loss1.clone().detach().cpu()) / 2,
                'f':
                fine_loss0.clone().detach().cpu() +
                fine_loss1.clone().detach().cpu(),
                'v_n':
                self.fine_v_num,
                # 'pv_n': self.fine_pv_num,
                'n':
                self.fine_all_num0 + self.fine_all_num1,
                'loss':
                loss.clone().detach().cpu(),
            }
            data.update({'loss': loss, 'loss_scalars': loss_scalars})
        else:
            loss = cas_loss  # + data['dummy_loss']  # .clone()
            loss_scalars = {
                'mf': cas_loss0.clone().detach().cpu(),
                'c': cas_loss0.clone().detach().cpu(),
                'f': cas_loss0.clone().detach().cpu(),
                'v_n': self.fine_v_num,
                # 'pv_n': self.fine_pv_num,
                'n': self.fine_all_num0 + self.fine_all_num1,
                'loss': loss.clone().detach().cpu(),
            }
            data.update({'loss': loss, 'loss_scalars': loss_scalars})


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['adamatcher']['loss']
        self.match_type = self.config['adamatcher']['match_coarse'][
            'match_type']
        self.sparse_spvs = self.config['adamatcher']['match_coarse'][
            'sparse_spvs']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """Point-wise CE / Focal Loss with 0 / 1 confidence as gt.

        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_pos_w = 0.0
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.0
            c_neg_w = 0.0

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert (not self.sparse_spvs
                    ), 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            loss_pos = -torch.log(conf[pos_mask])
            loss_neg = -torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']

            if self.sparse_spvs:
                pos_conf = (conf[:, :-1, :-1][pos_mask] if self.match_type
                            == 'sinkhorn' else conf[pos_mask])
                loss_pos = -alpha * torch.pow(1 - pos_conf,
                                              gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat(
                        [conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = -alpha * torch.pow(1 - neg_conf,
                                                  gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]

                loss = (c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                        if self.match_type == 'sinkhorn' else c_pos_w *
                        loss_pos.mean())
                return loss
                # positive and negative elements occupy similar promotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = (-alpha * torch.pow(1 - conf[pos_mask], gamma) *
                            (conf[pos_mask]).log())
                loss_neg = (-alpha * torch.pow(conf[neg_mask], gamma) *
                            (1 - conf[neg_mask]).log())
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller proportion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(
                type=self.loss_config['coarse_type']))

    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        correct_mask = torch.norm(expec_f_gt, p=float('inf'),
                                  dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if (
                    self.training
            ):  # this seldom happen when training, since we pad prediction with gt
                logger.warning(
                    'assign a false supervision to avoid ddp deadlock')
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] -
                      expec_f[correct_mask])**2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        # some gt matches might fall out of the fine-level window
        # correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        correct_mask = torch.norm(expec_f_gt, p=float('inf'),
                                  dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1.0 / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)
                  ).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if (
                    self.training
            ):  # this seldom happen during training, since we pad prediction with gt
                # sometimes there is not coarse-level gt at all.
                logger.warning(
                    'assign a false supervision to avoid ddp deadlock')
                correct_mask[0] = True
                weight[0] = 0.0
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] -
                      expec_f[correct_mask, :2])**2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    @torch.no_grad()
    def compute_c_weight(self, data):
        """compute element-wise weights for computing coarse-level loss."""
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] *
                        data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs
            and self.match_type == 'sinkhorn' else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight,
        )
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({'loss_c': loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({'loss_f': loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f':
                                 torch.tensor(1.0)})  # 1 is the upper bound

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({'loss': loss, 'loss_scalars': loss_scalars})
