"""
Configuration manager - 配置管理模块
功能：加载和管理YAML配置文件
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "./configs/system.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        if self.config_path.exists():
            self.load()
        else:
            print(f"[ConfigManager] WARNING: Config file not found: {config_path}")
            self._load_defaults()
    
    def load(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            print(f"[ConfigManager] Loaded config from {self.config_path}")
        except Exception as e:
            print(f"[ConfigManager] ERROR loading config: {e}")
            self._load_defaults()
    
    def save(self, path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            path: 保存路径（None则使用原路径）
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"[ConfigManager] Saved config to {save_path}")
        except Exception as e:
            print(f"[ConfigManager] ERROR saving config: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点分隔的路径）
        
        Args:
            key_path: 配置键路径，如 "camera.ir.width"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        
        Args:
            key_path: 配置键路径
            value: 要设置的值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def _load_defaults(self):
        """加载默认配置"""
        self.config = {
            "camera": {
                "ir": {"device_id": 22, "width": 640, "height": 512, "fps": 50},
                "tv": {"device_id": 11, "width": 1920, "height": 1080, "fps": 25}
            },
            "model": {
                "ir": {"path": "./weights/IR.rknn", "img_size": [640, 640], "obj_thresh": 0.25},
                "tv": {"path": "./weights/TV.rknn", "img_size": [1088, 1088], "obj_thresh": 0.25}
            },
            "tracking": {
                "tracker_type": "CSRT",
                "frame_skip_ir": 10,
                "frame_skip_tv": 15
            },
            "osd": {"enabled": True, "language": "zh"},
            "runtime": {"mode": "both", "headless": False, "quiet": False}
        }
        print("[ConfigManager] Loaded default configuration")
    
    # 便捷访问方法
    @property
    def camera_ir(self) -> dict:
        """获取IR相机配置"""
        return self.get("camera.ir", {})
    
    @property
    def camera_tv(self) -> dict:
        """获取TV相机配置"""
        return self.get("camera.tv", {})
    
    @property
    def model_ir(self) -> dict:
        """获取IR模型配置"""
        return self.get("model.ir", {})
    
    @property
    def model_tv(self) -> dict:
        """获取TV模型配置"""
        return self.get("model.tv", {})
    
    @property
    def tracking_config(self) -> dict:
        """获取跟踪配置"""
        return self.get("tracking", {})
    
    @property
    def osd_config(self) -> dict:
        """获取OSD配置"""
        return self.get("osd", {})
    
    @property
    def runtime_config(self) -> dict:
        """获取运行时配置"""
        return self.get("runtime", {})
    
    def override_from_args(self, args):
        """
        从命令行参数覆盖配置
        
        Args:
            args: argparse解析的参数对象
        """
        # 运行时配置
        if hasattr(args, 'mode'):
            self.set("runtime.mode", args.mode)
        if hasattr(args, 'pattern'):
            self.set("runtime.use_pattern", args.pattern)
        if hasattr(args, 'headless'):
            self.set("runtime.headless", args.headless)
        if hasattr(args, 'quiet'):
            self.set("runtime.quiet", args.quiet)
        if hasattr(args, 'max_frames'):
            self.set("runtime.max_frames", args.max_frames)
        
        # 模型配置
        if hasattr(args, 'ir_model'):
            self.set("model.ir.path", args.ir_model)
        if hasattr(args, 'tv_model'):
            self.set("model.tv.path", args.tv_model)
        if hasattr(args, 'ir_img_size'):
            self.set("model.ir.img_size", args.ir_img_size)
        if hasattr(args, 'tv_img_size'):
            self.set("model.tv.img_size", args.tv_img_size)
        
        # 跟踪配置
        if hasattr(args, 'frame_skip_ir'):
            self.set("tracking.frame_skip_ir", args.frame_skip_ir)
        if hasattr(args, 'frame_skip_tv'):
            self.set("tracking.frame_skip_tv", args.frame_skip_tv)
        
        # RTSP配置
        if hasattr(args, 'rtsp_ir'):
            self.set("rtsp.enabled", True)
            self.set("rtsp.ir_mount", args.rtsp_ir)
        if hasattr(args, 'rtsp_tv'):
            self.set("rtsp.enabled", True)
            self.set("rtsp.tv_mount", args.rtsp_tv)
        if hasattr(args, 'rtsp_encoder'):
            self.set("rtsp.encoder", args.rtsp_encoder)
        if hasattr(args, 'rtsp_bitrate'):
            self.set("rtsp.bitrate", args.rtsp_bitrate)
        
        # CSV配置
        if hasattr(args, 'save_csv'):
            self.set("csv_logging.merged_path", args.save_csv)
        if hasattr(args, 'save_csv_ir'):
            self.set("csv_logging.ir_path", args.save_csv_ir)
        if hasattr(args, 'save_csv_tv'):
            self.set("csv_logging.tv_path", args.save_csv_tv)
        
        print("[ConfigManager] Configuration overridden from command line arguments")
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "=" * 60)
        print("Current Configuration:")
        print("=" * 60)
        print(yaml.dump(self.config, default_flow_style=False, allow_unicode=True))
        print("=" * 60 + "\n")


# 测试代码
if __name__ == "__main__":
    # 测试配置管理器
    config = ConfigManager()
    
    # 测试获取配置
    print(f"IR Camera Width: {config.get('camera.ir.width')}")
    print(f"TV Model Path: {config.get('model.tv.path')}")
    print(f"Tracking Type: {config.get('tracking.tracker_type', 'CSRT')}")
    
    # 测试设置配置
    config.set("tracking.frame_skip_ir", 15)
    print(f"Updated IR frame skip: {config.get('tracking.frame_skip_ir')}")
    
    # 测试便捷访问
    print(f"\nCamera IR Config: {config.camera_ir}")
    print(f"Runtime Config: {config.runtime_config}")
    
    # 打印完整配置
    config.print_config()
