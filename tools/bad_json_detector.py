import os, json
from tqdm import tqdm

root = "/home/tsinghuaair/pengyh/FreeAskWorld"
bad_files = []

# 先统计所有 json 文件路径
json_files = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith(".json"):
            json_files.append(os.path.join(dirpath, fn))

# 遍历时加进度条
for fp in tqdm(json_files, desc="Scanning JSON files"):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:  # 捕获所有异常，保证不会中断
        bad_files.append((fp, str(e)))

# 扫描完成后统一输出坏文件
if bad_files:
    print("\n=== Bad JSON files detected ===")
    for fp, err in bad_files:
        print(f"[BAD] {fp}: {err}")

    with open("bad_files.txt", "w", encoding="utf-8") as f:
        for fp, err in bad_files:
            f.write(f"{fp}\t{err}\n")

print(f"\nTotal bad files: {len(bad_files)}")
