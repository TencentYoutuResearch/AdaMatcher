from configs.data.base import cfg

# TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info" # "./datasets/megadepth/index" # "assets/megadepth_test_1500_scene_info"
TEST_BASE_PATH = "./datasets/megadepth_test_1500_scene_info"
# TEST_BASE_PATH = "./datasets/megadepth/index/scene_info_scale_test_1109"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "./datasets/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = (
    f"{TEST_BASE_PATH}"  # f"{TEST_BASE_PATH}/scene_info_0.1_0.7" # f"{TEST_BASE_PATH}"
)
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"  # f"assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt"  # f"{TEST_BASE_PATH}/megadepth_test_1500.txt"
# cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_4_inf.txt"
# cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_2_4.txt"
# cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/test_1_2.txt"

# cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500_scale.txt"

# cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 640  # 1152  # 832 # 840 # 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
