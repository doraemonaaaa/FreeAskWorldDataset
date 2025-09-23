from tqdm import tqdm
import numpy as np
from FreeAskWorldDataset import FreeAskWorldDataset
# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset 实例
faw = FreeAskWorldDataset(root_folder)
faw.track_all_3d_box_velocity()