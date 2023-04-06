from configs.data.base import cfg

TEST_BASE_PATH = "./datasets/megadepth_scale_data/scale_data_0125"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "./datasets/2020visuallocalization/ImageMatching"
cfg.DATASET.TEST_NPZ_ROOT = (
    f"{TEST_BASE_PATH}"  # f"{TEST_BASE_PATH}/scene_info_0.1_0.7" # f"{TEST_BASE_PATH}"
)
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/phototourism_0.1.txt"


# cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 832  # 840 # 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
