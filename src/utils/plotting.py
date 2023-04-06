import bisect
import os
import pdb

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange


def _compute_conf_thresh(data):
    dataset_name = data["dataset_name"][0].lower()
    if dataset_name == "scannet":
        thr = 5e-4
    elif dataset_name == "megadepth":
        thr = 1e-4
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return thr


# --- VISUALIZATION --- #


def make_matching_fine(
    img0, img1, patch0_center_coord, kpts1, patch1_center_coord, kpts0, patch_size, path
):
    """
    Args:
        patch0_center_coord: [k, 2] <x, y>
        kpts1: [k, 2] <x, y>
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    levels = range(2, 256, 2)
    h, w, _ = img0.shape
    box_img0 = img0.copy()
    masked_img0 = img0.copy()
    box_img1 = img1.copy()
    masked_img1 = img1.copy()
    # axes[0, 1].imshow(img1.round().astype(np.int32))
    # axes[1, 0].imshow(img0.round().astype(np.int32))

    # colors_list0 = []
    patch0_l = patch0_center_coord[:, 1] * w + patch0_center_coord[:, 0]
    for i in range(0, h * w, 4):
        ids0_mask = patch0_l == i

        # colors_list0.append(color)
        d_kps1 = kpts1[ids0_mask]
        if len(d_kps1) == 0:
            continue
        color = np.array([np.random.choice(levels) for _ in range(3)])
        # axes[0, 1].scatter(d_kps1[:, 0], d_kps1[:, 1], c=np.array([color * 0.6/255]*len(d_kps1)), s=1)
        masked_img1[
            d_kps1[:, 1], d_kps1[:, 0]
        ] = color  # masked_img1[d_kps1[:,1], d_kps1[:, 0]]*0.4 + color * 0.6
        masked_img1[d_kps1[:, 1] - 1, d_kps1[:, 0]] = color
        masked_img1[d_kps1[:, 1] + 1, d_kps1[:, 0]] = color
        masked_img1[d_kps1[:, 1], d_kps1[:, 0] - 1] = color
        masked_img1[d_kps1[:, 1], d_kps1[:, 0] + 1] = color
        x0 = i % w - patch_size  # //2
        x1 = x0 + patch_size * 2
        y0 = i // w - patch_size  # //2
        y1 = y0 + patch_size * 2
        box_img0[y0:y1, x0:x1] = img0[y0:y1, x0:x1] * 0.4 + color * 0.6

    # colors_list1 = []
    patch1_l = patch1_center_coord[:, 1] * w + patch1_center_coord[:, 0]
    for j in range(0, img1.shape[0] * img1.shape[1], 4):
        ids1_mask = patch1_l == j
        # colors_list1.append(color)

        d_kps0 = kpts0[ids1_mask]
        if len(d_kps0) == 0:
            continue

        color = np.array([np.random.choice(levels) for _ in range(3)])
        # axes[1, 0].scatter(d_kps0[:, 0], d_kps0[:, 1], c=np.array([color * 0.6/255]*len(d_kps0)), s=1)
        masked_img0[
            d_kps0[:, 1], d_kps0[:, 0]
        ] = color  # masked_img0[d_kps0[:,1], d_kps0[:, 0]]*0.4 + color * 0.6
        masked_img0[d_kps0[:, 1] - 1, d_kps0[:, 0]] = color
        masked_img0[d_kps0[:, 1] + 1, d_kps0[:, 0]] = color
        masked_img0[d_kps0[:, 1], d_kps0[:, 0] - 1] = color
        masked_img0[d_kps0[:, 1], d_kps0[:, 0] + 1] = color
        x0 = j % w - patch_size
        x1 = x0 + patch_size * 2
        y0 = j // w - patch_size
        y1 = y0 + patch_size * 2
        box_img1[y0:y1, x0:x1] = img1[y0:y1, x0:x1] * 0.4 + color * 0.6

    axes[0, 0].imshow(box_img0.round().astype(np.int32))
    axes[0, 1].imshow(masked_img1.round().astype(np.int32))
    axes[1, 1].imshow(box_img1.round().astype(np.int32))
    axes[1, 0].imshow(masked_img0.round().astype(np.int32))
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def make_gt_matching_mask(batch, main_path):
    for b_id in range(batch["image0"].size(0)):
        img0 = (
            (batch["image0"][b_id].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )  # .astype(np.int32)
        img1 = (
            (batch["image1"][b_id].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )
        # class_k0_l0_ids = batch['spv_class_k0_l0_ids'][batch['spv_class_b0_l0_ids']==b_id]
        # class_k1_l0_ids = batch['spv_class_k1_l0_ids'][batch['spv_class_b1_l0_ids']==b_id]
        class_k0_l0_ids = batch["train_class_k0_l0_ids"][
            batch["train_class_b0_l0_ids"] == b_id
        ]
        class_k1_l0_ids = batch["train_class_k1_l0_ids"][
            batch["train_class_b1_l0_ids"] == b_id
        ]
        mask0 = batch["spv_instance_masks0"][b_id, class_k1_l0_ids].bool().cpu().numpy()
        mask1 = batch["spv_instance_masks1"][b_id, class_k0_l0_ids].bool().cpu().numpy()
        class_mask0 = batch["spv_class_l1_gt0"][b_id, 1].float().cpu().numpy() * 255
        class_mask1 = batch["spv_class_l1_gt1"][b_id, 1].float().cpu().numpy() * 255
        path = main_path + "_{}.jpg".format(b_id)
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
            draw_kpts=True,
        )


def make_pred_matching_mask(batch, main_path):
    for b_id in range(batch["image0"].size(0)):
        img0 = (
            (batch["image0"][b_id].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )  # .astype(np.int32)
        img1 = (
            (batch["image1"][b_id].cpu().numpy() * 255)
            .round()
            .astype(np.float32)
            .transpose(1, 2, 0)
        )  # .astype(np.int32)
        if "train_class_k0_l0_ids" in batch:
            class_k0_l0_ids = batch["train_class_k0_l0_ids"][
                batch["train_class_b0_l0_ids"] == b_id
            ]
            class_k1_l0_ids = batch["train_class_k1_l0_ids"][
                batch["train_class_b1_l0_ids"] == b_id
            ]
            class_b0_l0_ids = batch["train_class_b0_l0_ids"][
                batch["train_class_b0_l0_ids"] == b_id
            ]
            class_b1_l0_ids = batch["train_class_b1_l0_ids"][
                batch["train_class_b1_l0_ids"] == b_id
            ]
            instance_mask0 = (
                batch["conf_matrix0_d"][class_b1_l0_ids, class_k1_l0_ids]
                * batch["uncalculate_mask0"][batch["train_class_b1_l0_ids"] == b_id]
            )
            instance_mask1 = (
                batch["conf_matrix1_d"][class_b0_l0_ids, class_k0_l0_ids]
                * batch["uncalculate_mask1"][batch["train_class_b0_l0_ids"] == b_id]
            )
            mask0 = (
                rearrange(
                    (instance_mask0[class_b1_l0_ids == b_id] > 0.5).detach(),
                    "n (h w) -> n h w",
                    h=80,
                    w=80,
                )
                .cpu()
                .numpy()
            )
            mask1 = (
                rearrange(
                    (instance_mask1[class_b0_l0_ids == b_id] > 0.5).detach(),
                    "n (h w) -> n h w",
                    h=80,
                    w=80,
                )
                .cpu()
                .numpy()
            )
        else:
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
                * batch["uncalculate_mask0"][batch["pred_class_b1_l0_ids"] == b_id]
            )
            instance_mask1 = (
                batch["conf_matrix1_d"][class_b0_l0_ids, class_k0_l0_ids]
                * batch["uncalculate_mask1"][batch["pred_class_b0_l0_ids"] == b_id]
            )

            mask0 = instance_mask0[class_b1_l0_ids == b_id]
            mask0 = (
                rearrange((mask0 > 0.5).detach(), "n (h w) -> n h w", h=80, w=80)
                .cpu()
                .numpy()
            )
            mask1 = instance_mask1[class_b0_l0_ids == b_id]
            mask1 = (
                rearrange((mask1 > 0.5).detach(), "n (h w) -> n h w", h=80, w=80)
                .cpu()
                .numpy()
            )
        class_mask0 = batch["pred_class0_l0"][b_id].float().cpu().numpy() * 255
        class_mask1 = batch["pred_class1_l0"][b_id].float().cpu().numpy() * 255
        path = main_path + "_{}.jpg".format(b_id)
        # path = './viz/pred/pred_epoch{}_{}_{}.jpg'.format(self.current_epoch, self.count, b_id)
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
            draw_kpts=True,
        )


def make_matching_mask(
    img0,
    img1,
    mask0,
    mask1,
    class_mask0,
    class_mask1,
    path,
    ind0=None,
    ind1=None,
    dpi=300,
    draw_kpts=True,
):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=dpi)
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    levels = range(2, 256, 2)

    class_mask0 = cv2.resize(
        class_mask0.astype(np.float32), (img0.shape[1], img0.shape[0])
    )
    class_mask1 = cv2.resize(
        class_mask1.astype(np.float32), (img1.shape[1], img1.shape[0])
    )
    axes[1, 2].imshow(class_mask1.round().astype(np.int32), cmap="gray")
    axes[0, 2].imshow(class_mask0.round().astype(np.int32), cmap="gray")

    if draw_kpts:
        axes[0, 1].imshow(img1.round().astype(np.int32))
        axes[1, 1].imshow(img0.round().astype(np.int32))

    box_img0 = img0.copy()
    masked_img0 = img0.copy()
    box_img1 = img1.copy()
    masked_img1 = img1.copy()

    # pdb.set_trace()
    if ind1 is None:
        ind1 = range(mask1.shape[0])
    for i, yx in enumerate(ind1):
        m1 = mask1[i]
        if m1.any():
            color = np.array([np.random.choice(levels) for _ in range(3)])
            # cx, cy = (yx % 10 + 0.5) * 64, (yx // 10 + 0.5) * 64
            x0, y0, x1, y1 = (
                (yx % 13) * 64,
                (yx // 13) * 64,
                (yx % 13 + 1) * 64,
                (yx // 13 + 1) * 64,
            )
            box_img0[y0:y1, x0:x1] = img0[y0:y1, x0:x1] * 0.5 + color * 0.5
            if draw_kpts:
                ky1, kx1 = np.nonzero(m1)
                ky1 *= img1.shape[0] // m1.shape[0]
                kx1 *= img1.shape[1] // m1.shape[1]
                axes[0, 1].scatter(kx1, ky1, s=0.1, color=color / 255)
            else:
                m1 = cv2.resize(
                    m1.astype(np.float32), (img1.shape[1], img1.shape[0])
                ).astype(np.bool)
                masked_img1[m1] = img1[m1] * 0.5 + color * 0.5  # color
    axes[0, 0].imshow(box_img0.round().astype(np.int32))
    if not draw_kpts:
        axes[0, 1].imshow(masked_img1.round().astype(np.int32))

    # pdb.set_trace()
    if ind0 is None:
        ind0 = range(mask0.shape[0])
    for i, yx in enumerate(ind0):
        m0 = mask0[i]
        if m0.any():
            color = np.array([np.random.choice(levels) for _ in range(3)])
            # cx, cy = (yx % 10 + 0.5) * 64, (yx // 10 + 0.5) * 64
            x0, y0, x1, y1 = (
                (yx % 13) * 64,
                (yx // 13) * 64,
                (yx % 13 + 1) * 64,
                (yx // 13 + 1) * 64,
            )
            box_img1[y0:y1, x0:x1] = img1[y0:y1, x0:x1] * 0.5 + color * 0.5
            if draw_kpts:
                ky0, kx0 = np.nonzero(m0)
                ky0 *= img0.shape[0] // m0.shape[0]
                kx0 *= img0.shape[1] // m0.shape[1]
                axes[1, 1].scatter(kx0, ky0, s=0.1, color=color / 255)
            else:
                m0 = cv2.resize(
                    m0.astype(np.float32), (img0.shape[1], img0.shape[0])
                ).astype(np.bool)
                masked_img0[m0] = img0[m0] * 0.5 + color * 0.5  # color
    axes[1, 0].imshow(box_img1.round().astype(np.int32))
    if not draw_kpts:
        axes[1, 1].imshow(masked_img0.round().astype(np.int32))

    # pdb.set_trace()
    # kpts0 = np.array(kpts0)*64
    # colors = np.array(colors)
    # axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c=colors, s=8)
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def make_keypoints(img0, gt_kpts0, kpts0, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    axes[0].imshow(img0.round().astype(np.int32))
    axes[1].imshow(img0.round().astype(np.int32))
    axes[0].scatter(gt_kpts0[:, 0], gt_kpts0[:, 1], c="r", s=0.1)
    axes[1].scatter(kpts0[:, 0], kpts0[:, 1], c="b", s=0.1)
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def make_matching_inliers(img0, img1, kpts0, kpts1, inliers_mask, scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    levels = range(2, 256, 2)

    in_kpts0 = kpts0[inliers_mask]
    in_kpts1 = kpts1[inliers_mask]
    out_kpts0 = kpts0[~inliers_mask]
    out_kpts1 = kpts1[~inliers_mask]

    c = (
        np.array(
            [[np.random.choice(levels) for _ in range(3)] for j in range(len(in_kpts0))]
        )
        / 255
    )
    axes[0].imshow(img0.round().astype(np.int32))
    axes[0].scatter(in_kpts0[:, 0], in_kpts0[:, 1], s=2, color=c)
    axes[0].scatter(out_kpts0[:, 0], out_kpts0[:, 1], s=2, color="r")
    axes[1].imshow(img1.round().astype(np.int32))
    axes[1].scatter(in_kpts1[:, 0], in_kpts1[:, 1], s=2, color=c)
    axes[1].scatter(out_kpts1[:, 0], out_kpts1[:, 1], s=2, color="r")

    print(min(scores[~inliers_mask]), max(scores[~inliers_mask]))
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def make_epipolar_plot(img0, img1, kpts0, kpts1, F_0to1, path):
    h, w, _ = img0.shape
    levels = range(2, 256, 2)
    color = (
        np.array(
            [[np.random.choice(levels) for _ in range(3)] for j in range(len(kpts0))]
        )
        / 255
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    axes[0].imshow(img0.round().astype(np.int32))
    # axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='r', s=1)
    axes[1].imshow(img1.round().astype(np.int32))
    # axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='r', s=1)

    plot_epipolar(kpts0, kpts1, F_0to1, w, h, color, lw=1, ps=3)
    plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
    plt.close()


def epipolar_point(l, w, h):
    p1 = [0, -l[2] / l[1]]
    p2 = [-l[2] / l[0], 0]
    p3 = [-(l[2] + l[1] * h) / l[0], h]
    p4 = [w, -(l[2] + l[0] * w) / l[1]]
    conners = [p1, p2, p3, p4]
    valid = []
    for kp in conners:
        if kp[0] >= 0 and kp[1] >= 0 and kp[0] <= w and kp[1] <= h:
            valid.append(kp)
    return valid


def plot_epipolar(mkpts0, mkpts1, F, w, h, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    lines0 = cv2.computeCorrespondEpilines(mkpts1.reshape(-1, 1, 2), 2, F).reshape(
        -1, 3
    )
    lines1 = cv2.computeCorrespondEpilines(mkpts0.reshape(-1, 1, 2), 1, F).reshape(
        -1, 3
    )
    kpts0 = [epipolar_point(l, w, h) for l in lines0]
    kpts0_0 = [kp[0] for kp in kpts0 if len(kp) == 2]
    kpts0_1 = [kp[1] for kp in kpts0 if len(kp) == 2]
    kpts1 = [epipolar_point(l, w, h) for l in lines1]
    kpts1_0 = [kp[0] for kp in kpts1 if len(kp) == 2]
    kpts1_1 = [kp[1] for kp in kpts1 if len(kp) == 2]

    fkpts0_0 = transFigure.transform(ax[0].transData.transform(kpts0_0))
    fkpts0_1 = transFigure.transform(ax[0].transData.transform(kpts0_1))
    fkpts1_0 = transFigure.transform(ax[1].transData.transform(kpts1_0))
    fkpts1_1 = transFigure.transform(ax[1].transData.transform(kpts1_1))

    fig.lines = [
        matplotlib.lines.Line2D(
            (fkpts0_0[i, 0], fkpts0_1[i, 0]),
            (fkpts0_0[i, 1], fkpts0_1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts0_0))
    ]
    fig.lines += [
        matplotlib.lines.Line2D(
            (fkpts1_0[i, 0], fkpts1_1[i, 0]),
            (fkpts1_0[i, 1], fkpts1_1[i, 1]),
            zorder=1,
            transform=fig.transFigure,
            c=color[i],
            linewidth=lw,
        )
        for i in range(len(kpts1_0))
    ]
    ax[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=ps)
    ax[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=ps)


def make_matching_figure(
    img0,
    img1,
    mkpts0,
    mkpts1,
    color,
    kpts0=None,
    kpts1=None,
    text=[],
    dpi=75,
    path=None,
):
    # draw image pair
    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=1,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01,
        0.99,
        "\n".join(text),
        transform=fig.axes[0].transAxes,
        fontsize=15,
        va="top",
        ha="left",
        color=txt_color,
    )

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha="dynamic"):
    b_mask = data["m_bids"] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data["image0"][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data["image1"][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data["mkpts0_f"][b_mask].cpu().numpy()
    kpts1 = data["mkpts1_f"][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if "scale0" in data:
        kpts0 = kpts0 / data["scale0"][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data["scale1"][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data["epi_errs"][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data["conf_matrix_gt"][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == "dynamic":
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f"#Matches {len(kpts0)}",
        f"Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}",
        f"Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}",
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1, color, text=text)
    return figure


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode="evaluation"):
    """Make matching figures for a batch.

    Args:
        data (Dict): a batch updated by PL_AdaMatcher.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ["evaluation", "confidence"]  # 'confidence'
    figures = {mode: []}
    for b_id in range(data["image0"].size(0)):
        if mode == "evaluation":
            fig = _make_evaluation_figure(
                data, b_id, alpha=config.TRAINER.PLOT_MATCHES_ALPHA
            )
        elif mode == "confidence":
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f"Unknown plot mode: {mode}")
    figures[mode].append(fig)
    return figures


def dynamic_alpha(
    n_matches, milestones=[0, 300, 1000, 2000], alphas=[1.0, 0.8, 0.4, 0.2]
):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]
    ) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1),
        0,
        1,
    )
