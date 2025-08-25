import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize(v):
    return v / np.linalg.norm(v)

def look_rotation(forward, up):
    """模仿Unity的LookRotation，返回旋转矩阵"""
    f = normalize(forward)
    r = normalize(np.cross(up, f))   # 右
    u = np.cross(f, r)               # 修正后的上
    return np.stack([r, u, f], axis=1)  # 列向量是坐标轴

# 相机 -> Unity
cam_to_unity = np.diag([1, -1, 1])

# Unity cubemap六面
cubemap_directions = {
    "+X": (np.array([ 1, 0, 0]), np.array([0, 1, 0])),
    "-X": (np.array([-1, 0, 0]), np.array([0, 1, 0])),
    "+Y": (np.array([0, 1, 0]), np.array([0, 0,-1])),
    "-Y": (np.array([0,-1, 0]), np.array([0, 0, 1])),
    "+Z": (np.array([0, 0, 1]), np.array([0, 1, 0])),
    "-Z": (np.array([0, 0,-1]), np.array([0, 1, 0])),
}

# 计算最终的相机系 -> Ego旋转
for face, (fwd, up) in cubemap_directions.items():
    R_cubemap_unity = look_rotation(fwd, up)
    R_final = R_cubemap_unity @ cam_to_unity
    print(f"=== {face} ===")
    print(R_final)
    print()