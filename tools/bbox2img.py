import os.path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from FreeAskWorldDataset import FreeAskWorldDataset


def world_to_cam(box_translation, box_rotation, pose_translation, pose_rotation, cs_translation, cs_rotation):
    """
    将世界坐标系下的 box 映射到相机坐标系
    box_translation: 世界坐标系下的平移 np.array([x,y,z])
    box_rotation: 世界坐标系下的旋转四元数 [x,y,z,w]
    pose_translation: ego->world 平移
    pose_rotation: ego->world 四元数
    cs_translation: cam->ego 平移
    cs_rotation: cam->ego 四元数
    """
    # world -> ego
    r_ego = R.from_quat(pose_rotation)
    r_ego_inv = r_ego.inv()
    t_ego = r_ego_inv.apply(box_translation - pose_translation)
    r_box = R.from_quat(box_rotation)
    r_box_ego = r_ego_inv * r_box

    # ego -> cam
    r_cam = R.from_quat(cs_rotation)
    r_cam_inv = r_cam.inv()
    t_cam = r_cam_inv.apply(t_ego - cs_translation)
    r_box_cam = r_cam_inv * r_box_ego

    return t_cam, r_box_cam.as_quat()


def process_single_sample(sample, camera_name, dataset, global_counter):
    """
    处理单个样本：读取图片、绘制3D框、保存结果
    """
    cam_paths = dataset.get_sample_cam_path(sample)
    if camera_name not in cam_paths or not cam_paths[camera_name]:
        print(f"样本 step{sample.id} 缺少 {camera_name} 图片，跳过...")
        return global_counter

    img_path = cam_paths[camera_name][0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取样本 step{sample.id} 的图片: {img_path}，跳过...")
        return global_counter

    # 相机内参
    camera_params = dataset.get_sample_camera_parameters(sample)
    camera_intrinsic = np.array(
        next(cam["intrinsic"] for cam in camera_params if cam["name"] == camera_name)
    )

    # 相机外参
    sensor_calibrations = dataset.get_sensor2ego()
    try:
        cam_calib = next(c for c in sensor_calibrations if c.name == camera_name)
    except StopIteration:
        print(f"未找到 {camera_name} 的校准参数，跳过样本 step{sample.id}...")
        return global_counter

    cs_translation = np.array(cam_calib.translation)
    cs_rotation = np.array(cam_calib.rotation)

    # ego -> world
    ego2global = dataset.get_ego2global(sample)
    pose_translation = np.array(ego2global["translation"])
    pose_rotation = np.array(ego2global["rotation"])

    # 3D框
    boxes_3d = dataset.get_sample_3d_box(sample)
    track_class = {"Car", "Human"}

    for box in boxes_3d:
        if box['labelName'] not in track_class:
            continue

        # 世界坐标 -> 相机坐标
        t_cam, r_cam_quat = world_to_cam(
            np.array(box['translation']),
            np.array(box['rotation']),
            pose_translation,
            pose_rotation,
            cs_translation,
            cs_rotation
        )

        # 3D框角点
        w, l, h = box['size']
        corners = np.array([
            [w/2, l/2, -h/2], [w/2, -l/2, -h/2],
            [-w/2, -l/2, -h/2], [-w/2, l/2, -h/2],
            [w/2, l/2, h/2], [w/2, -l/2, h/2],
            [-w/2, -l/2, h/2], [-w/2, l/2, h/2]
        ])
        r_box_cam = R.from_quat(r_cam_quat)
        rotated_corners = r_box_cam.apply(corners)
        rotated_corners += t_cam
        corners_3d = rotated_corners.T  # (3,8)

        # 可见性判断 (Unity 坐标系 z 前)
        if np.sum(corners_3d[2, :] > 0) < 4:
            continue

        # 投影到图像平面
        points = np.concatenate((corners_3d, np.ones((1, 8))), axis=0)
        points = camera_intrinsic @ points[:3, :]
        points /= points[2, :]
        box_img = points.astype(np.int32)

        # 如果投影完全在图像外，直接跳过
        if np.all((box_img[0, :] < 0) | (box_img[0, :] >= img.shape[1]) |
                  (box_img[1, :] < 0) | (box_img[1, :] >= img.shape[0])):
            continue

        print(f"显示box:{box['instanceId']} + {box['labelName']}")
        # print(camera_name, t_cam, r_cam_quat)

        # 绘制3D框
        color = (64, 128, 255)
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]
        for start, end in edges:
            if corners_3d[2, start] > 0 and corners_3d[2, end] > 0:
                if (0 <= box_img[0, start] < img.shape[1] and 0 <= box_img[1, start] < img.shape[0] and
                    0 <= box_img[0, end] < img.shape[1] and 0 <= box_img[1, end] < img.shape[0]):
                    cv2.line(img, (box_img[0, start], box_img[1, start]),
                             (box_img[0, end], box_img[1, end]), color, 1)

        # 绘制类别标签， 放在box的右下角
        label_pt = (box_img[0, 0], box_img[1, 0])
        if 0 <= label_pt[0] < img.shape[1] and 0 <= label_pt[1] < img.shape[0]:
            cv2.putText(img, box['labelName'], label_pt, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # 保存结果
    scene_name = f"scene_{sample.epoch}" if hasattr(sample, 'epoch') else "scene_unknown"
    output_filename = f"output_{scene_name}_step{sample.id}_cnt{global_counter}.jpg"
    cv2.imwrite(output_filename, img)
    print(f"已保存: {output_filename} | 样本: {scene_name}/step{sample.id} | 图片尺寸: {img.shape[:2]}")

    return global_counter + 1


if __name__ == "__main__":
    DATASET_ROOT = "/home/tsinghuaair/pengyh/FreeAskWorld"
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"数据集路径不存在: {DATASET_ROOT}")

    dataset = FreeAskWorldDataset(root_folder=DATASET_ROOT)
    print("数据集初始化完成")

    target_camera = "PanoCamera_front"
    print(f"目标相机: {target_camera}")

    all_scenes = dataset.get_scenes()
    if not all_scenes:
        raise FileNotFoundError("未找到任何场景，请检查数据集结构")
    print(f"共发现 {len(all_scenes)} 个场景")

    # 只选第一个场景
    scene = all_scenes[8]
    print(f"选定场景: {scene.epoch}/{scene.name}")

    scene_samples = dataset.get_scene_samples(scene)
    if not scene_samples:
        raise FileNotFoundError(f"场景 {scene.name} 下无样本")

    # 只取第36个样本
    sample = scene_samples[37]
    print(f"选定样本: step{sample.id}")

    global_img_counter = 0
    global_img_counter = process_single_sample(
        sample=sample,
        camera_name=target_camera,
        dataset=dataset,
        global_counter=global_img_counter
    )

    print("="*50)
    print(f"测试完成！共生成 {global_img_counter} 张输出图片")
