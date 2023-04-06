import os
import pdb
import pprint
import subprocess

# import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from einops.einops import rearrange
from loguru import logger

from src.adamatcher import AdaMatcher
from src.adamatcher.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine,
)
from src.losses.ada_loss import AdaMatcherLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.comm import all_gather, gather
from src.utils.metrics import aggregate_metrics  # compute_coarse_error,
from src.utils.metrics import compute_pose_errors, compute_symmetrical_epipolar_errors
from src.utils.misc import flattenList, lower_config
from src.utils.plotting import (  # make_matching_figure_color,; make_matching_inliers,
    make_matching_figures,
    make_matching_mask,
)
from src.utils.profiler import PassThroughProfiler

# from matplotlib import pyplot as plt


def get_gpu_memory_map(id=0):
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    return gpu_memory[id] / 1024.0


class PL_AdaMatcher(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        self.lr = self.config.TRAINER.TRUE_LR
        _config = lower_config(self.config)
        self.ada_cfg = lower_config(_config["adamatcher"])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )

        # Matcher: AdaMatcher
        self.matcher = AdaMatcher(config=_config["adamatcher"])
        self.mask_loss = AdaMatcherLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            weights = torch.load(pretrained_ckpt, map_location="cpu")["state_dict"]
            # self.matcher.load_state_dict({k.replace('matcher.', ''): v for k, v in weights.items()})
            self.load_state_dict(weights)
            logger.info(f"Load '{pretrained_ckpt}' as pretrained checkpoint")

        # Testing
        self.dump_dir = dump_dir
        self.count = 0
        self.max_memory = 0
        self.min_memory = 1e6
        self.all_time = 0.0
        self.metric_time = 0.0

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # learning rate warm up
        # pdb.set_trace()
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            # for pg in optimizer.param_groups:
            #     pg['lr'] = self.config.TRAINER.TRUE_LR
            warmup_step = self.config.TRAINER.WARMUP_STEP
            if self.trainer.global_step < warmup_step:
                if self.config.TRAINER.WARMUP_TYPE == "linear":
                    base_lr = (
                        self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                    )
                    lr = base_lr + (
                        self.trainer.global_step / self.config.TRAINER.WARMUP_STEP
                    ) * abs(self.config.TRAINER.TRUE_LR - base_lr)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr
                elif self.config.TRAINER.WARMUP_TYPE == "constant":
                    pass
                else:
                    raise ValueError(
                        f"Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}"
                    )

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def _trainval_inference(self, batch):

        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)

        with self.profiler.profile("AdaMatcher"):
            self.matcher(batch)

        self.count += 1
        # with self.profiler.profile("Compute fine supervision"):
        #     compute_supervision_fine(batch, self.config)

        with self.profiler.profile("Compute losses"):
            # self.loss(batch)
            self.mask_loss(batch)

    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(
                batch
            )  # compute epi_errs for each match
            compute_pose_errors(
                batch, self.config
            )  # compute R_errs, t_errs, pose_errs for each pair
            # compute_coarse_error(batch)

            rel_pair_names = list(zip(*batch["pair_names"]))
            bs = batch["image0"].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
                "epi_errs": [
                    batch["epi_errs"][batch["m_bids"] == b].cpu().numpy()
                    for b in range(bs)
                ],
                "R_errs": batch["R_errs"],
                "t_errs": batch["t_errs"],
                "inliers": batch["inliers"],
            }
            ret_dict = {"metrics": metrics}
        return ret_dict, rel_pair_names

    def training_step(self, batch, batch_idx):
        # pdb.set_trace()
        self._trainval_inference(batch)

        # logging
        if (
            self.trainer.global_rank == 0
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            # scalars
            for k, v in batch["loss_scalars"].items():
                self.logger.experiment.add_scalar(f"train/{k}", v, self.global_step)

            # net-params
            if self.config.ADAMATCHER.MATCH_COARSE.MATCH_TYPE == "sinkhorn":
                self.logger.experiment.add_scalar(
                    f"skh_bin_score",
                    self.matcher.coarse_matching.bin_score.clone().detach().cpu().data,
                    self.global_step,
                )

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING:
                compute_symmetrical_epipolar_errors(
                    batch
                )  # compute epi_errs for each match
                figures = make_matching_figures(
                    batch, self.config, self.config.TRAINER.PLOT_MODE
                )
                for k, v in figures.items():
                    self.logger.experiment.add_figure(
                        f"train_match/{k}", v, self.global_step
                    )

            for k, v in batch["loss_scalars"].items():
                self.log(k, v, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return {"loss": batch["loss"], "loss_scalars": batch["loss_scalars"]}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                "train/avg_loss_on_epoch", avg_loss, global_step=self.current_epoch
            )

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self._trainval_inference(batch)
        # if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
        #     for k, v in batch['loss_scalars'].items():
        #         self.log(k, v, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        ret_dict, _ = self._compute_metrics(batch)
        if 0:
            bs = batch["image0"].shape[0]
            for b_id in range(bs):
                img0 = (
                    (batch["image0"][b_id].cpu().numpy() * 255)
                    .round()
                    .astype(np.float32)
                    .transpose(1, 2, 0)
                )
                img1 = (
                    (batch["image1"][b_id].cpu().numpy() * 255)
                    .round()
                    .astype(np.float32)
                    .transpose(1, 2, 0)
                )
                name0, name1 = batch["pair_names"]
                name0 = name0[0].split("/")[-1].split(".")[0]
                name1 = name1[0].split("/")[-1].split(".")[0]
                if 0:
                    pdb.set_trace()
                    import pickle

                    overlap0 = batch["classification0"][b_id].cpu().numpy()
                    overlap1 = batch["classification1"][b_id].cpu().numpy()
                    attn_mask0 = batch["attn_mask0"][b_id].cpu().numpy()
                    attn_mask1 = batch["attn_mask1"][b_id].cpu().numpy()
                    # save_data = dict(
                    #     img0=img0,
                    #     img1=img1,
                    #     overlap0=overlap0,
                    #     overlap1=overlap1,
                    #     attn_mask0=attn_mask0,
                    #     attn_mask1=attn_mask1,
                    # )
                    path = "./viz/pkl/{}-{}.pkl".format(name0, name1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    pkl_file = open(path, "wb")
                    pkl_file.dump()

                if 0:  # '0015' in batch['scene_id'][0]:
                    class_k0_l0_ids = batch["spv_class_k0_l0_ids"][
                        batch["spv_class_b0_l0_ids"] == b_id
                    ]
                    class_k1_l0_ids = batch["spv_class_k1_l0_ids"][
                        batch["spv_class_b1_l0_ids"] == b_id
                    ]
                    mask0 = (
                        batch["spv_instance_masks0"][b_id, class_k1_l0_ids]
                        .bool()
                        .cpu()
                        .numpy()
                    )
                    mask1 = (
                        batch["spv_instance_masks1"][b_id, class_k0_l0_ids]
                        .bool()
                        .cpu()
                        .numpy()
                    )
                    mask0 = (
                        batch["spv_instance_masks0"][b_id, class_k1_l0_ids]
                        .bool()
                        .cpu()
                        .numpy()
                    )
                    mask1 = (
                        batch["spv_instance_masks1"][b_id, class_k0_l0_ids]
                        .bool()
                        .cpu()
                        .numpy()
                    )
                    class_mask0 = (
                        batch["spv_class_l1_gt0"][b_id].float().cpu().numpy() * 255
                    )
                    class_mask1 = (
                        batch["spv_class_l1_gt1"][b_id].float().cpu().numpy() * 255
                    )
                    path = "./viz/gt/{}-{}.jpg".format(name0, name1)
                    if not os.path.exists(path):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        make_matching_mask(
                            img0,
                            img1,
                            mask0,
                            mask1,
                            class_mask0,
                            class_mask1,
                            path,
                            ind0=class_k1_l0_ids,
                            ind1=class_k0_l0_ids,
                            dpi=300,
                            draw_kpts=False,
                        )

                    class_k0_l0_ids = batch["pred_class_k0_l0_ids"][
                        batch["pred_class_b0_l0_ids"] == b_id
                    ]
                    class_k1_l0_ids = batch["pred_class_k1_l0_ids"][
                        batch["pred_class_b1_l0_ids"] == b_id
                    ]
                    class_b0_l0_ids = batch["pred_class_b0_l0_ids"][
                        batch["pred_class_b0_l0_ids"] == b_id
                    ]
                    class_b1_l0_ids = batch["pred_class_b1_l0_ids"][
                        batch["pred_class_b1_l0_ids"] == b_id
                    ]
                    instance_mask0 = (
                        batch["conf_matrix0_d"][class_b1_l0_ids, class_k1_l0_ids]
                        * batch["uncalculate_mask0"][
                            batch["pred_class_b1_l0_ids"] == b_id
                        ]
                    )
                    instance_mask1 = (
                        batch["conf_matrix1_d"][class_b0_l0_ids, class_k0_l0_ids]
                        * batch["uncalculate_mask1"][
                            batch["pred_class_b0_l0_ids"] == b_id
                        ]
                    )
                    mask0 = instance_mask0[class_b1_l0_ids == b_id]
                    mask0 = (
                        rearrange((mask0 > 0.5), "n (h w) -> n h w", h=104, w=104)
                        .cpu()
                        .numpy()
                    )
                    mask1 = instance_mask1[class_b0_l0_ids == b_id]
                    mask1 = (
                        rearrange(
                            (mask1 > 0.5).detach(), "n (h w) -> n h w", h=104, w=104
                        )
                        .cpu()
                        .numpy()
                    )
                    class_mask0 = (
                        batch["pred_class0_l1"][b_id].float().cpu().numpy() * 255
                    )
                    class_mask1 = (
                        batch["pred_class1_l1"][b_id].float().cpu().numpy() * 255
                    )
                    path = "./viz/pred/{}-{}.jpg".format(name0, name1)
                    if not os.path.exists(path):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        make_matching_mask(
                            img0,
                            img1,
                            mask0,
                            mask1,
                            class_mask0,
                            class_mask1,
                            path,
                            ind0=class_k1_l0_ids,
                            ind1=class_k0_l0_ids,
                            dpi=300,
                            draw_kpts=False,
                        )
        # val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        # figures = {self.config.TRAINER.PLOT_MODE: []}
        # if batch_idx % val_plot_interval == 0:
        #     figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            "loss_scalars": batch["loss_scalars"],
            # 'figures': figures,
        }

    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = (
            [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        )
        multi_val_metrics = defaultdict(list)

        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very beginning of the training
            cur_epoch = self.trainer.current_epoch
            if (
                not self.trainer.resume_from_checkpoint
                and self.trainer.running_sanity_check
            ):
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o["loss_scalars"] for o in outputs]
            loss_scalars = {
                k: torch.stack(
                    flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                ).mean()
                for k in _loss_scalars[0]
            }
            # for k, v in loss_scalars.items():
            #     print(k, v)
            # for k, v in loss_scalars.items():
            #     loss_scalars[k] = torch.stack(loss_scalars[k]).mean()

            # 2. val metrics: dict of list, numpy
            _metrics = [o["metrics"] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            print(val_metrics_4tb)
            for thr in [5, 10, 20]:
                multi_val_metrics[f"auc@{thr}"].append(val_metrics_4tb[f"auc@{thr}"])

            # 3. figures
            # _figures = [o['figures'] for o in outputs]
            # figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    # mean_v = torch.stack(v).mean()
                    mean_v = v
                    self.logger.experiment.add_scalar(
                        f"val_{valset_idx}/avg_{k}", mean_v, global_step=cur_epoch
                    )

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
                    )

        #         for k, v in figures.items():
        #             if self.trainer.global_rank == 0:
        #                 for plot_idx, fig in enumerate(v):
        #                     self.logger.experiment.add_figure(
        #                         f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
        #     plt.close('all')
        # for key in val_metrics_4tb.keys():
        #     self.log(key, torch.tensor(np.mean(val_metrics_4tb[key])))
        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(
                f"auc@{thr}", torch.tensor(np.mean(multi_val_metrics[f"auc@{thr}"]))
            )
            # ckpt monitors on this
        # for k, v in loss_scalars.items():
        #     self.log(k, v)

    def test_step(self, batch, batch_idx):
        # with self.profiler.profile("AdaMatcher"):
        with self.profiler.profile("AdaMatcher"):
            # t0 = time.monotonic()
            self.matcher(batch)
            # self.all_time += time.monotonic() - t0
        # pdb.set_trace()
        # tmp_gpu_use = get_gpu_memory_map(1)
        # if self.min_memory > tmp_gpu_use:
        #     self.min_memory = tmp_gpu_use
        # if self.max_memory < tmp_gpu_use:
        #     self.max_memory = tmp_gpu_use
        # t1 = time.monotonic()
        ret_dict, rel_pair_names = self._compute_metrics(batch)
        # self.metric_time += time.monotonic() - t1

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                # keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                keys_to_save = {"mkpts0_f", "mkpts1_f", "scores", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        if "classification" not in key:
                            item[key] = batch[key][mask].cpu().numpy()
                        else:
                            item[key] = batch[key][b_id].cpu().numpy()
                    for key in [
                        "R_errs",
                        "t_errs",
                        "inliers",
                    ]:  # 'fp_scores', 'miss_scores']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "Ada_pred_eval", dumps)
        # print(self.matcher.bb_time/2000., self.matcher.ficas_time/2000., self.matcher.coarse_time/2000., self.matcher.refine_time/2000., self.matcher.all_t/2000., self.matcher.n, self.all_time/2000., self.metric_time/2000.)
        # print(self.min_memory, self.max_memory)
