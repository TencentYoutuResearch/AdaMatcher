import os
import pdb

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

npy_file = './dump/scale15/LoFTR_pred_eval.npy'
data = np.load(npy_file, allow_pickle=True).tolist()

num_matches = 0
for i in tqdm(range(len(data))):
    num_matches += len(data[i]['mkpts1_f'])

print(num_matches, len(data), num_matches / len(data))
