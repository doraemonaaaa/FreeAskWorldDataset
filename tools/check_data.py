import os
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset 实例
faw = FreeAskWorldDataset(root_folder)

samples = faw.get_all_samples()

missing_files = []

# for sample in samples:
#     scene = sample["scene"]
#     scene_path = scene.path
#     vln_data_path = os.path.join(scene_path, "VLNData", "VLNData.json")

#     if not os.path.exists(vln_data_path):
#         missing_files.append(vln_data_path)

# if missing_files:
#     print("以下 VLNData.json 缺失：")
#     for path in missing_files:
#         print(path)
# else:
#     print("所有 VLNData.json 文件都存在 ✅")
    
bad_samples = []

for sample in samples:
    cam_paths = faw.get_sample_cam_path(sample)
    num_cams = len(cam_paths)

    if num_cams != 6:  # 如果不是 6 个相机
        bad_samples.append({
            "scene": sample["scene"].name if hasattr(sample["scene"], "name") else str(sample["scene"]),
            "sample_id": sample["id"],
            "num_cams": num_cams,
            "cams_found": list(cam_paths.keys())
        })

print("=" * 60)
print(" get_sample_cam_path 检查结果 ")
print("-" * 60)
if bad_samples:
    print(f"  共发现 {len(bad_samples)} 个样本相机数量不为 6：")
    for item in bad_samples:
        print(f"  场景: {item['scene']}, 样本ID: {item['sample_id']}, "
                f"相机数量: {item['num_cams']}, 已找到相机: {item['cams_found']}")
else:
    print("  ✅ 所有样本的相机数量都是 6")
print("=" * 60)
