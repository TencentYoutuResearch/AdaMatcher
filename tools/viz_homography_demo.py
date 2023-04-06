import os
import pdb
import pickle
import sys

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from homography_utils import left_right_move, warp_image
from kornia.geometry.epipolar import numeric
from scipy.stats import truncnorm
from tqdm import tqdm

sys.path.append("..")
from src.utils.dataset import get_divisible_wh, get_resized_wh
from src.utils.metrics import symmetric_epipolar_distance


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(
        inp.shape[-2:]
    ), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None

    padded = np.ones((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype) * 255
    padded[: inp.shape[0], : inp.shape[1], :] = inp
    if ret_mask:
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[: inp.shape[0], : inp.shape[1]] = True

    return padded, mask


def read_img(path, resize=832, df=32):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = get_resized_wh(w, h, resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = np.array([w / w_new, h / h_new], dtype=np.float)
    scale_wh = np.array([w_new, h_new], dtype=np.float)

    pad_to = max(h_new, w_new)
    image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    return image, mask, scale, scale_wh


def perspective_transform_inv(pts, H):
    H_inv = np.linalg.pinv(H)
    # H_inv = np.linalg.inv(H)
    inv_pts = cv2.perspectiveTransform(pts.reshape(1, -1, 2), H_inv.astype(np.float32))[
        0
    ]
    return inv_pts


if __name__ == "__main__":
    save_dir = "./viz_demo_homography"
    os.makedirs(save_dir, exist_ok=True)
    ada_pkl_path = "./demo_homography/ada_res.pkl"
    loftr_pkl_path = "./datasets/demo_homography/loftr_res.pkl"
    # img_path0 = './datasets/megadepth/train/Undistorted_SfM/0008/images/4062183688_789b33f30e_o.jpg'
    # img_path1 = './datasets/megadepth/train/Undistorted_SfM/0008/images/2991074704_ae5ced7e38_o.jpg'
    img_path0 = "./datasets/megadepth/train/Undistorted_SfM/0015/images/3538480162_734b651167_o.jpg"
    img_path1 = "./datasets/megadepth/train/Undistorted_SfM/0015/images/570188204_952af377b3_o.jpg"
    image0, mask0, scale0, scale_wh0 = read_img(img_path0, resize=832, df=32)
    image1, mask1, scale1, scale_wh1 = read_img(img_path1, resize=832, df=32)
    info_npz_path = (
        "./datasets/megadepth_scale_data/scale_data_0125/megadepth_scale_23.npz"
    )
    all_info = np.load(info_npz_path, allow_pickle=True)
    index0 = all_info["image_paths"].index(
        img_path0.replace("./datasets/megadepth/train/", "")
    )
    index1 = all_info["image_paths"].index(
        img_path1.replace("./datasets/megadepth/train/", "")
    )
    K0 = all_info["intrinsics"][index0]
    K1 = all_info["intrinsics"][index1]
    T0 = all_info["poses"][index0]
    T1 = all_info["poses"][index1]
    T_0to1 = np.matmul(T1, np.linalg.inv(T0))

    Tx = numeric.cross_product_matrix(torch.tensor(T_0to1[:3, 3])[None])[0].numpy()
    E_mat = Tx @ T_0to1[:3, :3]
    epi_thr = 5e-4  # 1e-4

    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    pts0 = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    pts1 = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    shape0 = np.asarray([w0, h0], np.float32).reshape([1, 2])
    shape1 = np.asarray([w1, h1], np.float32).reshape([1, 2])
    pts0 *= shape0
    pts1 *= shape1
    # h_max, w_max = max(h0, h1), max(w0, w1)

    with open(ada_pkl_path, "rb") as f:
        ada_info = pickle.load(f)
    with open(loftr_pkl_path, "rb") as f:
        loftr_info = pickle.load(f)

    num_frams = len(ada_info)
    for i in tqdm(range(num_frams)):
        # pts1 = left_right_move(pts1, h1, w1, 1, 0.005)
        # if i % 2 == 1:
        #     pts1 = left_right_move(pts1, h1, w1, 1, 0.02 * (i + 0.25))
        # else:
        #     pts1 = left_right_move(pts1, h1, w1, 1, -0.02 * (i))
        # H = cv2.getPerspectiveTransform(
        #     pts1.astype(np.float32), pts0.astype(np.float32)
        # )
        # image1 = cv2.warpPerspective(
        #     image1,
        #     H,
        #     (w1, h1),
        #     flags=cv2.INTER_LINEAR,
        #     borderMode=cv2.BORDER_CONSTANT,
        #     borderValue=(255, 255, 255),
        # )
        image1, H = warp_image(image1, pts0, pts1, h1, w1, i)

        key_name = img_path1.split("/")[-1].replace(".jpg", "-{}".format(i))
        ada_data = ada_info[key_name]
        loftr_data = loftr_info[key_name]
        ada_img = np.concatenate([image0, image1], axis=1)
        loftr_img = np.concatenate([image0, image1], axis=1)

        ada_p0s, ada_p1s = ada_data["mkpts0_f"].copy(), ada_data["mkpts1_f"].copy()
        ada_p0s /= scale0[None]
        ada_p1s /= scale1[None]
        # ada_p1s_h = ada_p1s @ np.linalg.inv(H).transpose()
        ada_p1s_h = perspective_transform_inv(ada_p1s, H)

        ada_dis = symmetric_epipolar_distance(
            torch.tensor(ada_p0s * scale0[None]),
            torch.tensor(ada_p1s_h * scale1[None]),
            torch.tensor(E_mat),
            torch.tensor(K0),
            torch.tensor(K1),
        ).numpy()
        ada_inlier_mask = ada_dis < epi_thr
        ada_p0s = ada_p0s[ada_inlier_mask]
        ada_p1s = ada_p1s[ada_inlier_mask]
        print(
            "ada matches_num:{}, valid_num:{}, P:{:.2f}".format(
                len(ada_inlier_mask),
                sum(ada_inlier_mask),
                sum(ada_inlier_mask) / len(ada_inlier_mask),
            )
        )

        loftr_p0s, loftr_p1s = (
            loftr_data["mkpts0_f"].copy(),
            loftr_data["mkpts1_f"].copy(),
        )
        loftr_p0s /= scale0[None]
        loftr_p1s /= scale1[None]
        # loftr_p1s_h = loftr_p1s @ np.linalg.inv(H).transpose()
        loftr_p1s_h = perspective_transform_inv(loftr_p1s, H)
        loftr_dis = symmetric_epipolar_distance(
            torch.tensor(loftr_p0s * scale0[None]),
            torch.tensor(loftr_p1s_h * scale1[None]),
            torch.tensor(E_mat),
            torch.tensor(K0),
            torch.tensor(K1),
        ).numpy()
        loftr_inlier_mask = loftr_dis < epi_thr
        loftr_p0s = loftr_p0s[loftr_inlier_mask]
        loftr_p1s = loftr_p1s[loftr_inlier_mask]
        print(
            "loftr matches_num:{}, valid_num:{}, P:{:.2f}".format(
                len(loftr_inlier_mask),
                sum(loftr_inlier_mask),
                sum(loftr_inlier_mask) / len(loftr_inlier_mask),
            )
        )

        ada_p1s[:, 0] += w0
        loftr_p1s[:, 0] += w0
        ada_p0s = np.round(ada_p0s).astype(int)
        ada_p1s = np.round(ada_p1s).astype(int)
        loftr_p0s = np.round(loftr_p0s).astype(int)
        loftr_p1s = np.round(loftr_p1s).astype(int)
        ada_color = np.zeros((len(ada_p1s), 3))
        ada_color[:, 1] = 255
        loftr_color = np.zeros((len(loftr_p1s), 3))
        loftr_color[:, 1] = 255
        for j in range(len(ada_color)):
            cv2.line(
                ada_img,
                tuple(ada_p0s[j]),
                tuple(ada_p1s[j]),
                ada_color[j],
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        for j in range(len(loftr_color)):
            cv2.line(
                loftr_img,
                tuple(loftr_p0s[j]),
                tuple(loftr_p1s[j]),
                loftr_color[j],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        img_cat = np.concatenate(
            [
                ada_img,
                np.ones((20, ada_img.shape[1], 3), dtype=np.uint8) * 255,
                loftr_img,
            ],
            axis=0,
        )
        img_cat = cv2.cvtColor(img_cat, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, "{}.jpg".format(i)), img_cat)
