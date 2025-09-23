import os
import json
import re
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset 实例
faw = FreeAskWorldDataset(root_folder)

# 获取所有 epoches 和场景（这里以第一个 epoch 和第一个场景为例）
scenes_dict = faw.get_scenes()
first_scene = scenes_dict[0]
# 获取该场景的所有 sample id
sample_ids = faw.get_scene_sample_ids(first_scene)
sample_id = sample_ids[0]  # 取第一个 sample

sample = faw.get_sample(first_scene, sample_id)
# 获取该 sample 的所有 3D box
sample_cam_path = faw.get_sample_cam_path(sample)

# 输出结果
for c, paths in sample_cam_path.items():  # 遍历 key, value
    for p in paths:  # 遍历同一 camera 的所有图片
        print("camera:" + c + "  " + "path:" + p)
