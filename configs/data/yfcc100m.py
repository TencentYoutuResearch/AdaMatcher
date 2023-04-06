from configs.data.base import cfg

TEST_BASE_PATH = "./datasets/yfcc"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "./datasets/yfcc/yfcc100m"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"

cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/yfcc_test.txt"


# cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = None
cfg.DATASET.MGDPT_IMG_PAD = False
cfg.DATASET.MGDPT_DF = 16
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
