from configs.data.base import cfg

TRAIN_BASE_PATH = "./datasets/imc2022"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TRAIN_DATA_ROOT = TRAIN_BASE_PATH
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/train_npz"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_scene.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

TEST_BASE_PATH = "./datasets/megadepth/index"
cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "./datasets/megadepth/test"
cfg.DATASET.VAL_NPZ_ROOT = (
    cfg.DATASET.TEST_NPZ_ROOT
) = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.VAL_LIST_PATH = (
    cfg.DATASET.TEST_LIST_PATH
) = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0  # for both test and val

# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100  # 100

cfg.DATASET.MGDPT_IMG_RESIZE = 832  # for training on 11GB mem GPUs
