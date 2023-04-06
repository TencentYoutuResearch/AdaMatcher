import os
import pdb
import sys

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.utils.plotting import error_colormap, make_matching_figure


def get_resize_scale(w, h, resize, df):
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    scale = np.array((w / w_new, h / h_new))
    return w_new, h_new, scale


def pad_right(inp, pad_w_size):
    h, w, c = inp.shape
    padded = np.ones((h, pad_w_size, inp.shape[2]), dtype=inp.dtype) * 255
    padded[:h, :w, :] = inp
    return padded


def pad_bottom(inp, pad_h_size):
    h, w, c = inp.shape
    padded = np.ones((pad_h_size, w, inp.shape[2]), dtype=inp.dtype) * 255
    if (pad_h_size - h) // 2 > 1:
        rh = (pad_h_size - h) // 2
        padded[rh : rh + h, :w, :] = inp
    else:
        rh = 0
        padded[:h, :w, :] = inp
    return padded, rh


def pad_edge(inp, pad_h_size, pad_w_size):
    h, w, c = inp.shape
    if pad_w_size > 0:
        padded = np.ones((pad_h_size, pad_w_size, inp.shape[2]), dtype=inp.dtype) * 255
        padded[:h, :w, :] = inp
    else:
        padded = np.ones((pad_h_size, w, inp.shape[2]), dtype=inp.dtype) * 255
        padded[:h, :w, :] = inp
    return padded


def draw_text(img, point, text, color, drawType="custom"):
    fontScale = 1.5  # 1.2  # 1 # 0.4
    thickness = 3  # 5
    text_thickness = 3  # 5  # 1
    fontFace = cv2.FONT_ITALIC  # cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        # draw score value
        cv2.putText(
            img,
            str(text),
            (text_loc[0], text_loc[1] + baseline),
            fontFace,
            fontScale,
            color,
            text_thickness,
            8,
        )
        # img = Image.fromarray(img)
        # draw = ImageDraw.Draw(img)
        # fontStyle = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", text_thickness)
        # draw.text((text_loc[0], text_loc[1] + baseline), text, color, font=text_thickness)
        # img = np.array(img)

    elif drawType == "simple":
        cv2.putText(img, "%d" % (text), point, fontFace, 0.5, color)
    return img


def eval_matches(p1s, p2s, homography):
    # Compute the reprojection errors from im1 to im2
    # with the given the GT homography
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogeneous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist


img_f = open("./ft_local/img23.txt", "r")
img_list = [l.split()[0] for l in img_f.readlines()]

root_dir = "./datasets/megadepth/test"
so = 12  # 45 # 23 # 34 # 45
npy_file_m2o = (
    f"./datasets/ft_local/SATR_overlap_0205/dump/scale{so}/LoFTR_pred_eval.npy"
)
npy_file_loftr = f"./datasets/LoFTR-master-official/dump/scale{so}/LoFTR_pred_eval.npy"
npy_file_sg = f"./datasets/ft_local/superglue-pylightning-jizhi/dump/scale{so}/SuperGlue_pred_eval.npy"
in_thr = 1e-4  # 3 # 5
out_dir = f"./viz_3methods_{so}"
# out_dir = f'./viz_sup_{so}'
os.makedirs(out_dir, exist_ok=True)
data_m2o = np.load(npy_file_m2o, allow_pickle=True).tolist()
data_loftr = np.load(npy_file_loftr, allow_pickle=True).tolist()
data_sg = np.load(npy_file_sg, allow_pickle=True).tolist()

for i_ in tqdm(range(len(data_m2o))):
    for j_ in range(len(data_loftr)):
        for k_ in range(len(data_sg)):
            if (
                data_m2o[i_]["pair_names"] != data_loftr[j_]["pair_names"]
                or data_m2o[i_]["pair_names"] != data_sg[k_]["pair_names"]
            ):
                continue

            m2o_item = data_m2o[i_]
            loftr_item = data_loftr[j_]
            sg_item = data_sg[k_]

            pair_name = (
                loftr_item["pair_names"][0].replace("/", "_").split(".")[0]
                + "-"
                + loftr_item["pair_names"][1].replace("/", "_").split(".")[0]
            )
            # if pair_name+'.jpg' not in img_list:
            #     continue
            img_path1 = os.path.join(root_dir, loftr_item["pair_names"][0])
            img_path2 = os.path.join(root_dir, loftr_item["pair_names"][1])

            ore, lre, sre = m2o_item["R_errs"], loftr_item["R_errs"], sg_item["R_errs"]
            ote, lte, ste = m2o_item["t_errs"], loftr_item["t_errs"], sg_item["t_errs"]

            if not (
                ore < lre * 0.5
                and ore < sre * 0.5
                and ote < lte * 0.5
                and ote < ste * 0.5
            ):
                continue

            img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            h1, w1 = img1.shape[:2]
            img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            h2, w2 = img2.shape[:2]

            h_max, w_max = max(h1, h2), max(w1, w2)
            img1, rh1 = pad_bottom(img1, h_max)
            img2, rh2 = pad_bottom(img2, h_max)

            img_m2o = np.concatenate([img1, img2], axis=1)
            img_loftr = img_m2o.copy()
            img_sg = img_m2o.copy()

            m2o_p1s, m2o_p2s = m2o_item["mkpts0_f"], m2o_item["mkpts1_f"]
            loftr_p1s, loftr_p2s = loftr_item["mkpts0_f"], loftr_item["mkpts1_f"]
            sg_p1s, sg_p2s = sg_item["mkpts0_f"], sg_item["mkpts1_f"]
            m2o_p1s[:, 1] += rh1
            m2o_p2s[:, 1] += rh2
            loftr_p1s[:, 1] += rh1
            loftr_p2s[:, 1] += rh2
            sg_p1s[:, 1] += rh1
            sg_p2s[:, 1] += rh2

            dist_m2o = m2o_item["epi_errs"]
            dist_loftr = loftr_item["epi_errs"]
            dist_sg = sg_item["epi_errs"]

            # AdaMatcher ##################################################
            m2o_p1s = np.round(m2o_p1s).astype(int)
            m2o_p2s = np.round(m2o_p2s).astype(int)
            m2o_p2s[:, 0] += w1
            m2o_num_matches = len(m2o_p1s)
            m2o_num_inliners = len(m2o_p1s[dist_m2o <= in_thr])
            m2o_color = np.zeros((len(m2o_p1s), 3))
            m2o_color[dist_m2o <= in_thr, 1] = 255
            m2o_color[dist_m2o > in_thr, 0] = 255

            m2o_p1s_out = m2o_p1s[dist_m2o > in_thr]
            m2o_p1s_in = m2o_p1s[dist_m2o <= in_thr]
            m2o_p2s_out = m2o_p2s[dist_m2o > in_thr]
            m2o_p2s_in = m2o_p2s[dist_m2o <= in_thr]
            m2o_color_out = m2o_color[dist_m2o > in_thr]
            m2o_color_in = m2o_color[dist_m2o <= in_thr]

            for i in range(len(m2o_color_out)):
                cv2.line(
                    img_m2o,
                    tuple(m2o_p1s_out[i]),
                    tuple(m2o_p2s_out[i]),
                    m2o_color_out[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            for i in range(len(m2o_color_in)):
                cv2.line(
                    img_m2o,
                    tuple(m2o_p1s_in[i]),
                    tuple(m2o_p2s_in[i]),
                    m2o_color_in[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            m2o_txt = [
                f"AdaMatcher: pair-{m2o_num_matches}",
                f"P({m2o_num_inliners/(m2o_num_matches+1e-4)*100:.1f}%):{m2o_num_inliners}/{m2o_num_matches}",
                f"err_R:{m2o_item['R_errs']:.2f}, err_t:{m2o_item['t_errs']:.2f}",
            ]

            ####### adamatcher ################################
            loftr_p1s = np.round(loftr_p1s).astype(int)
            loftr_p2s = np.round(loftr_p2s).astype(int)
            loftr_p2s[:, 0] += w1
            loftr_num_matches = len(loftr_p1s)
            loftr_num_inliners = len(loftr_p1s[dist_loftr <= in_thr])
            loftr_color = np.zeros((len(loftr_p1s), 3))
            loftr_color[dist_loftr <= in_thr, 1] = 255
            loftr_color[dist_loftr > in_thr, 0] = 255

            loftr_p1s_out = loftr_p1s[dist_loftr > in_thr]
            loftr_p1s_in = loftr_p1s[dist_loftr <= in_thr]
            loftr_p2s_out = loftr_p2s[dist_loftr > in_thr]
            loftr_p2s_in = loftr_p2s[dist_loftr <= in_thr]
            loftr_color_out = loftr_color[dist_loftr > in_thr]
            loftr_color_in = loftr_color[dist_loftr <= in_thr]

            for i in range(len(loftr_color_out)):
                cv2.line(
                    img_loftr,
                    tuple(loftr_p1s_out[i]),
                    tuple(loftr_p2s_out[i]),
                    loftr_color_out[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            for i in range(len(loftr_color_in)):
                cv2.line(
                    img_loftr,
                    tuple(loftr_p1s_in[i]),
                    tuple(loftr_p2s_in[i]),
                    loftr_color_in[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            loftr_txt = [
                f"AdaMatcher: pair-{loftr_num_matches}",
                f"P({loftr_num_inliners/(loftr_num_matches+1e-4) * 100:.1f}%):{loftr_num_inliners}/{loftr_num_matches}",
                f"err_R:{loftr_item['R_errs']:.2f}, err_t:{loftr_item['t_errs']:.2f}",
            ]

            ####### sg ################################
            sg_p1s = np.round(sg_p1s).astype(int)
            sg_p2s = np.round(sg_p2s).astype(int)
            sg_p2s[:, 0] += w1
            sg_num_matches = len(sg_p1s)
            sg_num_inliners = len(sg_p1s[dist_sg <= in_thr])
            sg_color = np.zeros((len(sg_p1s), 3))
            sg_color[dist_sg <= in_thr, 1] = 255
            sg_color[dist_sg > in_thr, 0] = 255

            sg_p1s_out = sg_p1s[dist_sg > in_thr]
            sg_p1s_in = sg_p1s[dist_sg <= in_thr]
            sg_p2s_out = sg_p2s[dist_sg > in_thr]
            sg_p2s_in = sg_p2s[dist_sg <= in_thr]
            sg_color_out = sg_color[dist_sg > in_thr]
            sg_color_in = sg_color[dist_sg <= in_thr]

            for i in range(len(sg_color_out)):
                cv2.line(
                    img_sg,
                    tuple(sg_p1s_out[i]),
                    tuple(sg_p2s_out[i]),
                    sg_color_out[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            for i in range(len(sg_color_in)):
                cv2.line(
                    img_sg,
                    tuple(sg_p1s_in[i]),
                    tuple(sg_p2s_in[i]),
                    sg_color_in[i],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
            sg_txt = [
                f"SuperPoint+SuperGlue: pair-{sg_num_matches}",
                f"P({sg_num_inliners/(sg_num_matches+1e-4) * 100:.1f}%):{sg_num_inliners}/{sg_num_matches}",
                f"err_R:{sg_item['R_errs']:.2f}, err_t:{sg_item['t_errs']:.2f}",
            ]

            img_cat = np.concatenate(
                [
                    img_sg,
                    np.ones((img_m2o.shape[0], 20, 3), dtype=np.uint8) * 255,
                    img_loftr,
                    np.ones((img_m2o.shape[0], 20, 3), dtype=np.uint8) * 255,
                    img_m2o,
                ],
                axis=1,
            )
            fig, axes = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
            axes.imshow(img_cat)
            axes.axis("off")

            fig.text(
                0.01,
                0.99,
                "\n".join(sg_txt),
                transform=fig.axes[0].transAxes,
                fontsize=5,
                va="top",
                ha="left",
                color="k" if img_sg[:100, :200].mean() > 200 else "w",
            )
            fig.text(
                0.34,
                0.99,
                "\n".join(loftr_txt),
                transform=fig.axes[0].transAxes,
                fontsize=5,
                va="top",
                ha="left",
                color="k" if img_loftr[:100, :200].mean() > 200 else "w",
            )
            fig.text(
                0.68,
                0.99,
                "\n".join(m2o_txt),
                transform=fig.axes[0].transAxes,
                fontsize=5,
                va="top",
                ha="left",
                color="k" if img_m2o[:100, :200].mean() > 200 else "w",
            )

            path = os.path.join(out_dir, pair_name + ".jpg")
            plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
            plt.close()
            # pdb.set_trace()
