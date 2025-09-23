import os
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置数据集根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset
faw = FreeAskWorldDataset(root_folder)

faw.check_dataset()
