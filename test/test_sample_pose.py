import os
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorld 实例
faw = FreeAskWorldDataset(root_folder)

# 获取所有 epoches 和场景
scenes = faw.get_scenes()
first_scene = scenes[0]

# 获取该场景的所有 sample id
sample_ids = faw.get_scene_sample_ids(first_scene)

# 取第一个非起始 sample（避免 1 没有前一帧）
sample_id = sample_ids[2] if len(sample_ids) > 1 else sample_ids[0]

sample = faw.get_sample(first_scene, sample_id)
# 计算该 sample 的 pose 数据
pose_data = faw.get_sample_pose_data(sample)

# 输出结果
print(f"Scene: {first_scene}, Sample: {sample_id}")
print("Pose data:")
for k, v in pose_data.items():
    print(f"{k}: {v}")
