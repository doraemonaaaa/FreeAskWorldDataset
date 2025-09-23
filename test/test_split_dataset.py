import os
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset 实例
faw = FreeAskWorldDataset(root_folder)

# 调用 split_dataset
splits = faw.split_dataset()

# 输出结果
print("==== 数据集划分结果 ====")
for split_name, scenes in splits.items():
    print(f"{split_name}: {len(scenes)} 个场景")
    # 打印前 5 个，避免太长
    for scene in scenes[:5]:
        print(f"  - {scene}")
    if len(scenes) > 5:
        print("  ...")
