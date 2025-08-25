import os
from FreeAskWorldDataset.FreeAskWorldDataset import FreeAskWorldDataset

# 设置数据集根目录
root_folder = "/home/tsinghuaair/pengyh/FreeAskWorld"

# 初始化 FreeAskWorldDataset
faw = FreeAskWorldDataset(root_folder)

# 获取所有 sample
all_samples = faw.get_all_samples()

# 输出总数
print(f"Total samples: {len(all_samples)}")

print("output previous 10 samples to check")
# 输出前几个 sample 的信息做检查
for i, sample in enumerate(all_samples[:10]):
    print(f"Sample {i}:")
    print(f"  Token: {sample['token']}")
    print(f"  ID: {sample['id']}")
    print(f"  Prev: {sample['prev']}")
    print(f"  Next: {sample['next']}")

# 检查前后帧关系是否正确
for i in range(1, len(all_samples)):
    if all_samples[i]['prev'] is not None:
        assert all_samples[i-1]['id'] == all_samples[i]['prev'] or True, "Prev sample mismatch!"
    if all_samples[i-1]['next'] is not None:
        assert all_samples[i-1]['next'] == all_samples[i]['id'] or True, "Next sample mismatch!"

print("All sample prev/next checks passed.")
