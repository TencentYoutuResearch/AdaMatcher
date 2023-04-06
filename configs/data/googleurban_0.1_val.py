from configs.data.base import cfg

TEST_BASE_PATH = "./datasets/megadepth_scale_data/scale_data_0125"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "./datasets/2020visuallocalization/ImageMatching"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/googleurban_0.1.txt"


cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
