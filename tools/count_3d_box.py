import os
from FreeAskWorldDataset import FreeAskWorldDataset

# 设置根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset 实例
faw = FreeAskWorldDataset(root_folder)

def count_all_3dbox(dataset: FreeAskWorldDataset):
    total_boxes = 0
    samples = dataset.get_all_samples()

    for sample in samples:
        boxes = dataset.get_sample_3d_box(sample)
        total_boxes += len(boxes)

    print("=" * 60)
    print(" FreeAskWorld 数据集 3D Box 统计 ")
    print("-" * 60)
    print(f" 样本总数     : {len(samples)}")
    print(f" 3D Box 总数  : {total_boxes}")
    print(f" 平均每样本 box 数: {total_boxes / len(samples):.2f}")
    print("=" * 60)

    return total_boxes

# 调用
count_all_3dbox(faw)
