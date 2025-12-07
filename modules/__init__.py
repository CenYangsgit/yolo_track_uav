"""
Modules package for YOLO track UAV project
无人机目标检测跟踪项目 - 功能模块包
"""

__version__ = "2.0.0"

# 导出所有功能模块
from .config_manager import ConfigManager
from .serial_comm import SerialComm, TrackingStatus, TrackingData
from .osd_overlay import OSDOverlay
from .image_saver import ImageSaver
from .manual_selector import ManualSelector

__all__ = [
    "ConfigManager",
    "SerialComm",
    "TrackingStatus",
    "TrackingData",
    "OSDOverlay",
    "ImageSaver",
    "ManualSelector",
]
