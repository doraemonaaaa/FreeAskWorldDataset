from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
from typing import List, Dict, Tuple, Optional

@dataclass
class Sample:
    token: str
    scene: str
    id: int
    timestamp: float
    prev: Optional[int]
    next: Optional[int]
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass
class Scene:
    epoch: str
    name: str
    path: str
    
    def __init__(self, epoch, name, path):
        self.epoch = epoch
        self.name = name
        self.path = path
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass
class PoseData:
    position: np.ndarray
    rotation: np.ndarray       # 四元数 [x,y,z,w]
    translation_offset: np.ndarray
    quaternion_offset: np.ndarray
    euler_offset: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    time_offset: float = 0.0

    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass
class CameraParameters:
    name: str
    intrinsic: list           # 或 np.ndarray
    width: int
    height: int

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass
class SceneInfo:
    EpisodeID: str
    TrajectoryID: str
    SceneID: str
    TimeSpan: float
    StartTimeOfDay: float
    WeatherTypeName: str

    def __getitem__(self, key):
        return getattr(self, key)
    
    
@dataclass
class SensorCalibration:
    name: str
    translation: List[float]  # [x, y, z]
    rotation: List[float]     # [x, y, z, w]
    
    def __getitem__(self, key):
        return getattr(self, key)
    