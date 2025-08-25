import os
import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import hashlib
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from FreeAskWorldDataset.FreeAskWorldData import Sample, Scene, PoseData, CameraParameters, SceneInfo, SensorCalibration
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class FreeAskWorldDataset():
    sample_synthetic_data_type = ("Depth", "Normal", "instance segmentation", "semantic segmentation")
    sample_json_synthetic_data_type = ("instance segmentation", "Depth", "semantic segmentation", "Normal", "bounding box 3D", "bounding box")
    cameras = ("PanoCamera_back",  "PanoCamera_front", "PanoCamera_left", "PanoCamera_right", "PanoCamera_down", "PanoCamera_up")
    step_transform_deltatime = 1  # 1s, same to the time of sample
    semantic_categories = ("Car", "Human", "SideWalk", "Obstacle",
                           "TrafficSign", "Building", "StoreFront", 
                           "Bicycle", "Road", "RoadBlock", "Booth",
                           "Carts", "RoadMarking", "CrossWalk", "Parking",
                           "Advertising")
    instance_categories = ("Car", "Human", "Obstacle", "TrafficSign",
                            "Building", "Bicycle", "RoadBlock", "Booth"
                            "Carts", "Advertising")
    
    # ego and global using unity axis z/forward x/right y/up
    # ego = unity forward
    # camera using camera axis z/forward x/right y/down
    # lidar is not exist, so we set it to forward of ego
    # cam2ego
    # lidar2ego
    # ego2global
    
    ego_width = 0.4
    ego_length = 0.4
    ego_height = 1.8
    
    # cache
    all_samples = []
    

    def __init__(self, root_folder):
        self.root_folder = root_folder
        # for nuscenes format implementation
        self.fake_lidar_path = os.path.join(root_folder, "fake_lidar_4_dimension.bin")

    def _get_sample_synthetic_data(self, sample):
        scene = sample["scene"]
        sample_id = sample["id"]
        scene_path = scene.path
        sample_step = "step" + str(sample_id)
        samples_folder = os.path.join(scene_path, "PerceptionData", "solo", "sequence.0")

        if not os.path.exists(samples_folder):
            print(f"[WARN] Samples folder not found: {samples_folder}")
            return None
        
        for name in os.listdir(samples_folder):
            if sample_step in name and name.endswith(".json"):
                file_path = os.path.join(samples_folder, name)
                data = self._load_json(file_path)
                return data

        print(f"[WARN] No JSON file found for {sample_step} in {samples_folder}")
        return None
    
    def _write_sample_synthetic_data(self, sample, data):
        """
        Write modified synthetic JSON data back to the sample file.

        Args:
            sample: dict, the sample dict containing 'scene' and 'id'
            data: dict, the modified synthetic data to write
        """
        scene = sample["scene"]
        sample_id = sample["id"]
        scene_path = scene.path
        sample_step = "step" + str(sample_id)
        samples_folder = os.path.join(scene_path, "PerceptionData", "solo", "sequence.0")

        if not os.path.exists(samples_folder):
            raise FileNotFoundError(f"Samples folder not found: {samples_folder}")

        file_path = None
        for name in os.listdir(samples_folder):
            if sample_step in name and name.endswith(".json"):
                file_path = os.path.join(samples_folder, name)
                break
        if file_path is None:
            raise FileNotFoundError(f"No JSON file found for {sample_step} in {samples_folder}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
        
    def split_dataset(self):
        all_scenes = self.get_scenes()
        # 构建快速查找表，用 name 作为 key
        scene_map = {scene["name"]: scene for scene in all_scenes}
        
        split_json_path = os.path.join(self.root_folder, "DatasetSplit.json")
        split_data = self._load_json(split_json_path)

        # check is the scene exist
        for split_name, scene_ids in split_data.items():
            for scene_id in scene_ids:
                if scene_id not in scene_map:
                    print(f"[WARN] Scene {scene_id} not found in dataset root")
        
        result = {}
        for split_name, scene_ids in split_data.items():
            if split_name == "TOTAL":
                continue
            result[split_name] = []
            for scene_id in scene_ids:
                scene = scene_map[scene_id]  # 根据 name 得到完整 scene（包含 epoch）
                path = scene.path
                if path:   # 只加入存在的
                    result[split_name].append(path)
                else:
                    print(f"[WARN] Scene {scene_id} not found in dataset root")
        return result
    
    def check_dataset(self):
        """
        遍历所有 scene，检查 PerceptionData/solo/sequence.0 是否存在。
        如果缺少 solo 或 sequence.0，会列出来。
        """
        scenes = self.get_scenes()
        missing_scenes = []

        for scene in scenes:
            scene_path = scene.path
            perception_path = os.path.join(scene_path, "PerceptionData")
            solo_path = os.path.join(perception_path, "solo")
            sequence_path = os.path.join(solo_path, "sequence.0")

            if not os.path.exists(perception_path):
                missing_scenes.append((scene_path, "PerceptionData 缺失"))
            elif not os.path.exists(solo_path):
                missing_scenes.append((scene_path, "solo 缺失"))
            elif not os.path.exists(sequence_path):
                missing_scenes.append((scene_path, "sequence.0 缺失"))

        if missing_scenes:
            print("发现缺失的文件夹：")
            for path, reason in missing_scenes:
                print(f"{path} -> {reason}")
        else:
            print("所有 scene 的 PerceptionData/solo/sequence.0 文件夹都存在。")
    
    def get_epoches(self):
        epoches = []
        for name in os.listdir(self.root_folder):
            path = os.path.join(self.root_folder, name)
            if os.path.isdir(path) and name.lower() != "maps":  # 在这里过滤
                epoches.append(name)
        return epoches
    
    def get_scenes(self):
        scenes = []
        epoches = self.get_epoches()
        for epoch in epoches:
            epoch_path = os.path.join(self.root_folder, epoch)
            for name in os.listdir(epoch_path):
                path = os.path.join(epoch_path, name)
                if os.path.isdir(path):
                    scenes.append(Scene(epoch=epoch, name=name, path=path))
        return scenes
    
    def get_scene_poses(self, scene):
        scene_path = scene.path
        pose_json_path = os.path.join(scene_path, "PerceptionData", "step_transform.json")
        if not os.path.exists(pose_json_path):
            raise FileNotFoundError(f"Pose file not found: {pose_json_path}")
        poses = self._load_json(pose_json_path)
        return poses
    
    def get_scene_sample_ids(self, scene):
        scene_path = scene.path
        sample_ids = []
        seen = set()
        samples_folder = os.path.join(scene_path, "PerceptionData", "solo", "sequence.0")

        if not os.path.exists(samples_folder):
            raise FileNotFoundError(f"Samples folder not found: {samples_folder}")

        for name in os.listdir(samples_folder):
            # 匹配 stepN 的名字
            match = re.match(r"step(\d+)", name)
            if match:
                step_id = int(match.group(1))
                if step_id not in seen:
                    sample_ids.append(step_id)
                    seen.add(step_id)
        return sample_ids
    
    def get_sample(self, scene, sample_id, sample_ids=None,  idx=None):
        """
        获取某个 sample 的基本信息，包括 prev / next 指针和唯一 token
        """
        if sample_ids is None:
            sample_ids = sorted(self.get_scene_sample_ids(scene))
        if idx is None:  # 如果没传，就自己找（O(N)）
            idx = sample_ids.index(sample_id)

        prev_id = sample_ids[idx - 1] if idx > 0 else None
        next_id = sample_ids[idx + 1] if idx < len(sample_ids) - 1 else None

        token_str = f"{scene.path}_step{sample_id}"
        token = hashlib.md5(token_str.encode("utf-8")).hexdigest()

        return Sample(
            token=token,
            scene=scene,
            id=sample_id,
            timestamp=sample_id*self.step_transform_deltatime,
            prev=prev_id,
            next=next_id,
        )
        
    def get_scene_samples(self, scene):
        """
        获取某个场景下的所有 samples
        """
        samples = []
        sample_ids = sorted(self.get_scene_sample_ids(scene))
        for idx, sid in enumerate(sample_ids):
            sample = self.get_sample(scene, sid, sample_ids, idx)
            samples.append(sample)
        return samples

    def get_all_samples(self):
        self.all_samples = []
        scenes = self.get_scenes()
        for scene in scenes:
            sample_ids = sorted(self.get_scene_sample_ids(scene))
            for idx, sid in enumerate(sample_ids):
                sample_info = self.get_sample(scene, sid, sample_ids, idx)
                self.all_samples.append(sample_info)
        
        num_scenes = len(scenes)
        num_samples = len(self.all_samples)
        avg_samples = num_samples / num_scenes if num_scenes > 0 else 0
        print("=" * 60)
        print(" FreeAskWorld 数据集样本统计 ")
        print("-" * 60)
        print(f"  场景数量     : {num_scenes}")
        print(f"  样本总数     : {num_samples}")
        print(f"  平均每场景样本数 : {avg_samples:.2f}")
        print("=" * 60)
        return self.all_samples
    
    def get_sample_cam_path(self, sample) -> Dict[str, List[str]]:
        scene = sample["scene"]
        sample_id = sample["id"]
        scene_path = scene.path
        samples_folder = os.path.join(scene_path, "PerceptionData", "solo", "sequence.0")
        sample_step = f"step{sample_id}."

        if not os.path.exists(samples_folder):
            raise FileNotFoundError(f"Samples folder not found: {samples_folder}")

        # cameras 名称正则
        camera_pattern = "|".join(re.escape(cam) for cam in self.cameras)
        # synthetic 类型正则
        dtype_pattern = "|".join(re.escape(dtype) for dtype in self.sample_synthetic_data_type)

        # 正则匹配：以 step{sample_id}. 开头，包含 camera 名称，.png 结尾
        pattern = re.compile(rf"^{re.escape(sample_step)}.*({camera_pattern}).*\.png$", re.IGNORECASE)

        camera_pngs_dict = defaultdict(list)
        for name in os.listdir(samples_folder):
            match = pattern.match(name)
            if not match:
                continue
            if dtype_pattern and re.search(dtype_pattern, name, re.IGNORECASE):
                continue

            camera_name = match.group(1)  # 正则捕获的 camera 名称
            camera_pngs_dict[camera_name].append(os.path.join(samples_folder, name))

        return dict(camera_pngs_dict)  # 转成普通 dic
        
    # 坐标位于世界坐标系
    def get_sample_3d_box(self, sample):
        synthetic_json_data = self._get_sample_synthetic_data(sample)
        if synthetic_json_data is None:
            return [] 
        
        captures = synthetic_json_data["captures"]
        seen_box_instance_id = set()
        box_3ds = []

        for capture in captures:
            annotations = capture.get("annotations", [])
            for ann in annotations:
                if "bounding box 3D" in ann.get("id", ""):
                    box3d_values = ann.get("values", [])
                    for box in box3d_values:
                        instance_id = box.get("instanceId")
                        if instance_id not in seen_box_instance_id:
                            seen_box_instance_id.add(instance_id)
                            box_3ds.append(box)
        return box_3ds
    
    def track_all_3d_box(self, debug=True):
        '''
        process the dataset, write the next or prev to synthetic data json's 3dbox content
        3d box data example:
        {
            'instanceId': 325, 
            'labelId': 8, 
            'labelName': 'Booth', 
            'translation': [-10.8339958, -1.43815529, 16.8094521], 
            'size': [0.9983427, 2.56662488, 1.00039852], 
            'rotation': [0.0, 1.00000012, 0.0, 1.55943349e-06], 
            'velocity': [0.0, 0.0, 0.0], 
            'acceleration': [0.0, 0.0, 0.0]
        }
        '''
        samples = self.get_all_samples()

        for sample in tqdm(samples, desc="Tracking 3D boxes"):
            scene = sample["scene"]
            
            # 当前帧数据
            cur_data = self._get_sample_synthetic_data(sample)
            if cur_data is None:
                if debug:
                    print(f"[WARN] No JSON found for sample {sample['id']}, skipping...")
                continue  # 跳过当前 sample，继续下一个
            
            # 前一帧
            prev_dict = {}
            if sample["prev"] is not None:
                sample_prev = self.get_sample(scene, sample["prev"], self.get_scene_sample_ids(scene))
                prev_boxes = self.get_sample_3d_box(sample_prev)
                prev_dict = {b["instanceId"]: b for b in prev_boxes} if prev_boxes else {}

            # 后一帧
            next_dict = {}
            if sample["next"] is not None:
                sample_next = self.get_sample(scene, sample["next"], self.get_scene_sample_ids(scene))
                next_boxes = self.get_sample_3d_box(sample_next)
                next_dict = {b["instanceId"]: b for b in next_boxes} if next_boxes else {}

            # 遍历当前帧 box
            for capture in cur_data.get("captures", []):
                for ann in capture.get("annotations", []):
                    if "bounding box 3D" in ann.get("id", ""):
                        for box in ann.get("values", []):
                            iid = box["instanceId"]
                            has_prev = iid in prev_dict
                            has_next = iid in next_dict

                            if debug:
                                print(f"[Sample {sample['id']}] Box {iid}: "
                                    f"prev={'✔' if has_prev else '✘'}, "
                                    f"next={'✔' if has_next else '✘'}")
                                
                            # 创建浅拷贝并去掉 prev/next
                            def shallow_box_copy(box):
                                return {k: v for k, v in box.items() if k not in ("prev", "next")}

                            box["prev"] = shallow_box_copy(prev_dict[iid]) if iid in prev_dict else None
                            box["next"] = shallow_box_copy(next_dict[iid]) if iid in next_dict else None
                            
            # 写回 JSON
            self._write_sample_synthetic_data(sample, cur_data)
            if debug:
                print(f"✔ Sample {sample['id']} updated.")
        
    def get_sample_pose_data(self, sample) -> PoseData:
        scene = sample["scene"]
        sample_id = sample["id"]
        poses_datas = self.get_scene_poses(scene)  # 加载 json 数据
        dt = self.step_transform_deltatime
        
        if str(sample_id) not in poses_datas:
            print(f"[Warning] Pose for sample {sample_id} not found, using default values.")
            return PoseData(
                position=np.zeros(3),
                rotation=np.array([0, 0, 0, 1]),
                translation_offset=np.zeros(3),
                quaternion_offset=np.array([0, 0, 0, 1]),
                euler_offset=np.zeros(3),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
                time_offset=self.step_transform_deltatime
            )

        # 当前帧
        pose_curr = poses_datas[str(sample_id)]
        t_curr = np.array(pose_curr["position"])
        q_curr = np.array(pose_curr["rotation"])  # [x,y,z,w]

        #print(f"[DEBUG] t_curr: {t_curr}")
        #print(f"[DEBUG] q_curr: {q_curr}")

        result = {
            "position": t_curr,
            "rotation": q_curr,
            "translation_offset": np.zeros(3),
            "quaternion_offset": np.array([0, 0, 0, 1]),
            "euler_offset": np.zeros(3),
            "linear_velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "linear_acceleration": np.zeros(3),
            "time_offset": self.step_transform_deltatime
        }

        # 如果有上一帧，可以计算 offset、速度和角速度
        if sample_id >= 1 and str(sample_id - 1) in poses_datas:
            pose_prev = poses_datas[str(sample_id - 1)]
            t_prev = np.array(pose_prev["position"])
            q_prev = np.array(pose_prev["rotation"])
            #print(f"[DEBUG] t_prev: {t_prev}")
            #print(f"[DEBUG] q_prev: {q_prev}")

            # --- 平移 offset ---
            translation_offset = t_curr - t_prev
            #print(f"[DEBUG] translation_offset: {translation_offset}")
            result["translation_offset"] = translation_offset

            # --- 旋转 offset ---
            r_prev = R.from_quat(q_prev)
            r_curr = R.from_quat(q_curr)
            r_delta = r_prev.inv() * r_curr
            quaternion_offset = r_delta.as_quat()
            #print(f"[DEBUG] quaternion_offset: {quaternion_offset}")
            result["quaternion_offset"] = quaternion_offset
            result["euler_offset"] = R.from_quat(quaternion_offset).as_euler('xyz', degrees=True)  # using degree

            # --- 线速度 ---
            v = translation_offset / dt
            #print(f"[DEBUG] linear_velocity: {v}")
            result["linear_velocity"] = v

            # --- 角速度 ---
            angle = r_delta.magnitude()
            if angle != 0:
                axis = r_delta.as_rotvec() / angle
            else:
                axis = np.zeros(3)
            angular_velocity = axis * (angle / dt)
            #print(f"[DEBUG] angular_velocity: {angular_velocity}")
            result["angular_velocity"] = angular_velocity

            # 如果有再前一帧，可以算加速度
            if sample_id >= 2 and str(sample_id - 2) in poses_datas:
                pose_pre2 = poses_datas[str(sample_id - 2)]
                t_pre2 = np.array(pose_pre2["position"])
                #print(f"[DEBUG] t_prev2: {t_pre2}")
                v_prev = (t_prev - t_pre2) / dt
                a = (v - v_prev) / dt
                #print(f"[DEBUG] linear_acceleration: {a}")
                result["linear_acceleration"] = a
        else:
            # 第一帧或上一帧数据不存在，设置默认值
            result["translation_offset"] = np.zeros(3)
            result["quaternion_offset"] = np.array([0, 0, 0, 1])  # 单位四元数
            result["euler_offset"] = np.zeros(3)
            result["linear_velocity"] = np.zeros(3)
            result["angular_velocity"] = np.zeros(3)
            result["linear_acceleration"] = np.zeros(3)

        return PoseData(**result)

    def get_sample_camera_parameters(self, sample) -> List[CameraParameters]:
        scene = sample["scene"]
        sample_id = sample["id"]
        scene_path = scene.path
        sythetic_data = self._get_sample_synthetic_data(sample)
        sensors = sythetic_data["captures"]
        params_all = []
        for s in sensors:
            sensor_id = s["id"]
            camera_name = None
            for cam in self.cameras:
                if sensor_id.startswith(cam):
                    camera_name = cam
                    break
            if camera_name == None:
                raise ValueError(f"Can't find sensor")
            params_all.append(CameraParameters(
                name=camera_name,
                intrinsic=s["matrix"],
                width=s["dimension"][0],
                height=s["dimension"][1]
            ))
        return params_all
    
    # lidar and camera
    def get_sensor2ego(self):
        calibrated_sensor_json_path = os.path.join(self.root_folder, "calibrated_sensor.json")
        
        calibrations = self._load_json(calibrated_sensor_json_path)
        results = []
        for calibration in calibrations:
            sensor = SensorCalibration(
                name=calibration["name"],
                translation=calibration["translation"],
                rotation=calibration["rotation"]
            )
            results.append(sensor)
        return results

    def get_ego2global(self, sample):
        pose_data = self.get_sample_pose_data(sample)
        result = {
            "translation": pose_data.position,   # ndarray
            "rotation": pose_data.rotation       # ndarray
        }
        return result
    
    def get_scene_info(self, scene) -> SceneInfo:
        """
        获取某个 scene 的基本信息（从 VLNData.json 读取）
        返回一个 dict
        """
        scene_path = scene.path
        vln_data_path = os.path.join(scene_path, "VLNData", "VLNData.json")
        data = self._load_json(vln_data_path) 

        return SceneInfo(
            EpisodeID=data.get("EpisodeID", ""),
            TrajectoryID=data.get("TrajectoryID", ""),
            SceneID=data.get("SceneID", ""),
            TimeSpan=data.get("TimeSpan", 0.0),
            StartTimeOfDay=data.get("StartTimeOfDay", 0.0),
            WeatherTypeName=data.get("WeatherTypeName", "")
        )