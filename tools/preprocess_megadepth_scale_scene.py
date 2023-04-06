import os
import pdb
import random

import h5py
import numpy as np
from tqdm import tqdm

random.seed(66)
np.random.seed(66)


def boxes(points):
    box = np.array([points[0].min(), points[1].min(), points[0].max(), points[1].max()])
    return box


def overlap_box(K1, depth1, pose1, K2, depth2, pose2):
    mask1 = np.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    X1 = (u1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (v1 - K1[1, 2]) * (Z1 / K1[1, 1])
    XYZ1_hom = np.concatenate(
        [
            X1.reshape(1, -1),
            Y1.reshape(1, -1),
            Z1.reshape(1, -1),
            np.ones_like(Z1.reshape(1, -1)),
        ],
        axis=0,
    )
    XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

    uv2_hom = K2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
    h, w = depth2.shape
    i = uv2[0, :].astype(int)
    j = uv2[1, :].astype(int)

    valid_corners = np.logical_and(
        np.logical_and(i >= 0, j >= 0), np.logical_and(i < w, j < h)
    )

    valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
    valid_uv2 = uv2[:, valid_corners].astype(int)
    # depth validation
    Z2 = depth2[valid_uv2[1], valid_uv2[0]]
    inlier_mask = np.absolute(XYZ2[2, valid_corners] - Z2) < 1.0

    valid_uv1 = valid_uv1[:, inlier_mask]
    valid_uv2 = valid_uv2[:, inlier_mask]
    if valid_uv1.shape[1] == 0 or valid_uv2.shape[1] == 0:
        return np.array([0] * 4), np.array([0] * 4)

    box1 = boxes(valid_uv1)
    box2 = boxes(valid_uv2)
    return box1, box2


def scale_diff(bbox0, bbox1, depth0, depth1):
    # w_diff = max((bbox0[2] - bbox0[0])/(bbox1[2] - bbox1[0]), (bbox1[2] - bbox1[0])/(bbox0[2] - bbox0[0]))
    # h_diff = max((bbox0[3] - bbox0[1])/(bbox1[3] - bbox1[1]), (bbox1[3] - bbox1[1])/(bbox0[3] - bbox0[1]))
    image_h_scale = max(
        depth0.shape[0] / (bbox0[3] - bbox0[1]), depth1.shape[0] / (bbox1[3] - bbox1[1])
    )
    image_w_scale = max(
        depth0.shape[1] / (bbox0[2] - bbox0[0]), depth1.shape[1] / (bbox1[2] - bbox1[0])
    )

    if depth0.shape[0] / (bbox0[3] - bbox0[1]) > depth1.shape[0] / (
        bbox1[3] - bbox1[1]
    ):
        h_index = 0
        image_h_scale = depth0.shape[0] / (bbox0[3] - bbox0[1])
    else:
        h_index = 1
        image_h_scale = depth1.shape[0] / (bbox1[3] - bbox1[1])

    if depth0.shape[1] / (bbox0[2] - bbox0[0]) > depth1.shape[1] / (
        bbox1[2] - bbox1[0]
    ):
        w_index = 0
        image_w_scale = depth0.shape[1] / (bbox0[2] - bbox0[0])
    else:
        w_index = 1
        image_w_scale = depth1.shape[1] / (bbox1[2] - bbox1[0])

    if image_h_scale > image_w_scale:
        return image_h_scale, h_index
    else:
        return image_w_scale, w_index


if __name__ == "__main__":

    num_2_4 = 300
    num_4_inf = 450
    scene_list = ["0015", "0022"]
    scale_ratio = ["4_inf"]  # ['2_4', '4_inf']
    root_dir = "./datasets/megadepth/train/"
    scene_info_root = "./datasets/megadepth/index/scene_info_val_1500"
    save_dir = "./datasets/LoFTR_0910_sf/assets/megadepth_test_1500_scene_info"

    # pdb.set_trace()
    for scene_name in scene_list:
        for s_r in scale_ratio:

            npz_path = os.path.join(scene_info_root, scene_name + "_" + s_r + ".npz")
            if s_r == "2_4":
                num = num_2_4
            else:
                num = num_4_inf
            save_path = os.path.join(
                save_dir, scene_name + "_" + s_r + "_{}.npz".format(2)
            )
            scene_info = np.load(npz_path, allow_pickle=True)
            # 'image_paths', 'depth_paths', 'intrinsics', 'poses', 'pair_infos'
            pair_infos = list(scene_info["pair_infos"])
            image_paths = scene_info["image_paths"]
            depth_paths = scene_info["depth_paths"]
            intrinsics = scene_info["intrinsics"]
            poses = scene_info["poses"]

            scale_score_list = []
            scale_index = []
            for idx in tqdm(range(len(pair_infos))):
                (idx0, idx1), overlap_score, central_matches = pair_infos[idx]
                img_name0 = os.path.join(root_dir, image_paths[idx0])
                img_name1 = os.path.join(root_dir, image_paths[idx1])

                depth_path0 = os.path.join(root_dir, depth_paths[idx0])
                with h5py.File(depth_path0, "r") as hdf5_file:
                    depth0 = np.array(hdf5_file["/depth"])

                depth_path1 = os.path.join(root_dir, depth_paths[idx1])
                with h5py.File(depth_path1, "r") as hdf5_file:
                    depth1 = np.array(hdf5_file["/depth"])

                K0, K1 = intrinsics[idx0], intrinsics[idx1]
                pose0, pose1 = poses[idx0], poses[idx1]

                bbox0, bbox1 = overlap_box(K0, depth0, pose0, K1, depth1, pose1)
                if bbox0.max() > 0 and bbox1.max() > 0:
                    score, index = scale_diff(bbox0, bbox1, depth0, depth1)
                    scale_score_list.append(score)
                    scale_index.append(index)
                else:
                    pdb.set_trace()

            # 'image_paths', 'depth_paths', 'intrinsics', 'poses', 'pair_infos'
            len_ = len(pair_infos)
            pdb.set_trace()
            print("len_{}:".format(s_r), scene_name, len_)
            np.savez_compressed(
                save_path,
                image_paths=image_paths,
                depth_paths=depth_paths,
                intrinsics=intrinsics,
                poses=poses,
                pair_infos=pair_infos,  # random.sample(pair_infos, num),
                scale_scores=np.array(scale_score_list),
                scale_index=np.array(scale_index),
            )
