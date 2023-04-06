import os
import pdb
import pickle

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from homography_utils import (forward_backward_move, left_right_move,
                              nearest_identity, perspective_pts, rotate_move,
                              rotate_pts, sample_homography, scale_move,
                              scale_pts, translate_pts, up_down_move,
                              warp_image)
from scipy.stats import truncnorm

from src.adamatcher import AdaMatcher
from src.config.default import get_cfg_defaults
from src.utils.dataset import read_megadepth_color
from src.utils.misc import lower_config


def prepare_data(img_path0, img_path1, df, img_padding=True):
    image0, mask0, scale0, scale_wh0 = read_megadepth_color(
        img_path0, 832, df, img_padding, None
    )

    image1, mask1, scale1, scale_wh1 = read_megadepth_color(
        img_path1, 832, df, img_padding, None
    )

    [mask0_d8, mask1_d8] = F.interpolate(
        torch.stack([mask0, mask1], dim=0)[None].float(),
        scale_factor=1 / 8,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    [mask0_l0, mask1_l0] = F.interpolate(
        torch.stack([mask0, mask1], dim=0)[None].float(),
        scale_factor=1 / df,
        mode="nearest",
        recompute_scale_factor=False,
    )[0].bool()
    return (
        image0,
        mask0,
        scale0,
        scale_wh0,
        mask0_d8,
        mask0_l0,
        image1,
        mask1,
        scale1,
        scale_wh1,
        mask1_d8,
        mask1_l0,
    )


if __name__ == "__main__":
    name = "ada"  # ['ada', 'loftr']
    device = torch.device("cuda:0")
    config = get_cfg_defaults()
    _config = lower_config(config)
    matcher = AdaMatcher(config=_config["adamatcher"])
    ckpt_path = "weights/adamatcher.ckpt"
    weights = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict({k.replace("matcher.", ""): v for k, v in weights.items()})
    matcher = matcher.to(device)
    matcher.eval()

    save_dir = "./demo_homography"
    img_path0 = "./datasets/megadepth/train/Undistorted_SfM/0015/images/3538480162_734b651167_o.jpg"
    img_path1 = "./datasets/megadepth/train/Undistorted_SfM/0015/images/570188204_952af377b3_o.jpg"
    # img_path0 = './datasets/megadepth/train/Undistorted_SfM/0008/images/4062183688_789b33f30e_o.jpg'
    # img_path1 = './datasets/megadepth/train/Undistorted_SfM/0008/images/2991074704_ae5ced7e38_o.jpg'
    """
    Undistorted_SfM_0032_images_2684419109_f079e20d1b_o-Undistorted_SfM_0032_images_3171344491_2e1a4ac323_o
    Undistorted_SfM_0032_images_3517248137_8a63294eda_o-Undistorted_SfM_0032_images_3801727134_b5b0285a3f_o
    """
    df = 32
    (
        image0,
        mask0,
        scale0,
        scale_wh0,
        mask0_d8,
        mask0_l0,
        image1,
        mask1,
        scale1,
        scale_wh1,
        mask1_d8,
        mask1_l0,
    ) = prepare_data(img_path0, img_path1, df=df)

    h0, w0 = image0.shape[1:]
    h1, w1 = image1.shape[1:]
    pts0 = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    pts1 = np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    shape0 = np.asarray([w0, h0], np.float32).reshape([1, 2])
    shape1 = np.asarray([w1, h1], np.float32).reshape([1, 2])
    pts0 *= shape0
    pts1 *= shape1

    image1 = (image1 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    frame_list = []
    save_results = dict()
    for i in range(10):

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

        # mask1_d8 = mask1_d8.numpy().astype(np.uint8)
        # mask1_d8 = cv2.warpPerspective(mask1_d8, H, (w1//8, h1//8), flags=cv2.INTER_LINEAR)
        # mask1_d8 = torch.tensor(mask1_d8)
        # mask1_l0 = mask1_l0.numpy().astype(np.uint8)
        # mask1_l0 = cv2.warpPerspective(mask1_l0, H, (w1//df, h1//df), flags=cv2.INTER_LINEAR)
        # mask1_l0 = torch.tensor(mask1_l0)

        data = {
            "image0": image0.unsqueeze(0).to(device),  # (3, h, w)
            "image1": (
                torch.from_numpy(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)).float() / 255
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device),
            "mask0_d8": mask0_d8.unsqueeze(0).to(device),
            "mask1_d8": mask1_d8.unsqueeze(0).to(device).bool(),
            "mask0_l0": mask0_l0.unsqueeze(0).to(device),
            "mask1_l0": mask1_l0.unsqueeze(0).to(device).bool(),
            "scale0": scale0.unsqueeze(0).to(device),
            "scale1": scale1.unsqueeze(0).to(device),
            "scale_wh0": scale_wh0.unsqueeze(0).to(device),
            "scale_wh1": scale_wh1.unsqueeze(0).to(device),
        }
        with torch.no_grad():
            matcher(data)
        print(data["mkpts0_f"].shape)
        save_results[img_path1.split("/")[-1].replace(".jpg", "-{}".format(i))] = dict(
            scale0=scale0.numpy(),
            scale1=scale1.numpy(),
            scale_wh0=scale_wh0.numpy(),
            scale_wh1=scale_wh1.numpy(),
            mkpts0_f=data["mkpts0_f"].cpu().numpy(),
            mkpts1_f=data["mkpts1_f"].cpu().numpy(),
            scores=data["scores"].cpu().numpy(),
            h_matrix=H,
        )
        cv2.imwrite(os.path.join(save_dir, "{}_{}.jpg".format(name, i)), image1)
        # frame_list.append(image1)

    # gif = imageio.mimsave('./homographt.gif', frame_list+frame_list[::-1], 'GIF', duration=0.85)
    # gif = imageio.mimsave('./homographt.gif', frame_list, 'GIF', duration=1.0)
    with open(os.path.join(save_dir, "{}_res.pkl".format(name)), "wb") as f:
        pickle.dump(save_results, f)
    print("finish")
