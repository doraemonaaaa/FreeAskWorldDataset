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
# 传感器到 ego
sensor2ego_list = faw.get_sensor2ego()
print(f"Scene: {first_scene}, Sample: {sample_id}")
print("Sensor -> Ego transforms:")
for s in sensor2ego_list:
    print(f"  {s['name']}:")
    print(f"    translation: {s['translation']}")
    print(f"    rotation:    {s['rotation']}")

# ego 到 global
ego2global = faw.get_ego2global(sample)
print("\nEgo -> Global transform:")
print(f"  translation: {ego2global['translation']}")
print(f"  rotation:    {ego2global['rotation']}")


# 获取相机参数
camera_params = faw.get_sample_camera_parameters(sample)

print("\nCamera parameters:")
for cam in camera_params:
    print(f"  Camera: {cam.name}")
    print(f"    width: {cam.width}, height: {cam.height}")
    print(f"    intrinsic matrix: {cam.intrinsic}")