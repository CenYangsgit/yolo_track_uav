"""
Image saver module - 图像截图保存模块
功能：支持手动触发和定时自动保存截图
"""

import os
import cv2
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class ImageSaver:
    """图像保存器 - 支持手动和自动截图"""
    
    def __init__(self, 
                 save_dir: str = "./results/snapshots",
                 auto_save: bool = False,
                 auto_interval: int = 60,
                 max_images: int = 1000):
        """
        初始化图像保存器
        
        Args:
            save_dir: 保存目录
            auto_save: 是否启用自动保存
            auto_interval: 自动保存间隔（秒）
            max_images: 最大保存图片数量
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.auto_interval = auto_interval
        self.max_images = max_images
        
        self.last_auto_save = time.time()
        self.save_counter = 0
        
        # 为IR和TV创建子目录
        (self.save_dir / "IR").mkdir(exist_ok=True)
        (self.save_dir / "TV").mkdir(exist_ok=True)
        
        print(f"[ImageSaver] initialized at {self.save_dir}")
        print(f"[ImageSaver] auto_save={auto_save}, interval={auto_interval}s")
    
    def save_manual(self, 
                   frame, 
                   prefix: str = "manual",
                   subdir: str = "") -> Optional[str]:
        """
        手动保存图像
        
        Args:
            frame: 图像帧（numpy array）
            prefix: 文件名前缀
            subdir: 子目录（如 "IR" 或 "TV"）
            
        Returns:
            保存的文件路径，失败返回None
        """
        if frame is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}.jpg"
        
        save_path = self.save_dir / subdir / filename if subdir else self.save_dir / filename
        
        try:
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.save_counter += 1
            print(f"[ImageSaver] saved: {save_path}")
            
            # 检查是否超过最大数量
            self._cleanup_old_images(subdir)
            
            return str(save_path)
        except Exception as e:
            print(f"[ImageSaver] ERROR saving image: {e}")
            return None
    
    def check_auto_save(self, 
                       frame, 
                       prefix: str = "auto",
                       subdir: str = "") -> Optional[str]:
        """
        检查并执行自动保存
        
        Args:
            frame: 图像帧
            prefix: 文件名前缀
            subdir: 子目录
            
        Returns:
            如果保存了则返回路径，否则返回None
        """
        if not self.auto_save:
            return None
        
        current_time = time.time()
        if current_time - self.last_auto_save >= self.auto_interval:
            self.last_auto_save = current_time
            return self.save_manual(frame, prefix=prefix, subdir=subdir)
        
        return None
    
    def save_ir_tv_pair(self, 
                       frame_ir, 
                       frame_tv,
                       prefix: str = "pair") -> Tuple[Optional[str], Optional[str]]:
        """
        同时保存IR和TV图像对
        
        Args:
            frame_ir: 红外图像
            frame_tv: 可见光图像
            prefix: 文件名前缀
            
        Returns:
            (ir_path, tv_path) 元组
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        ir_path = None
        tv_path = None
        
        if frame_ir is not None:
            ir_filename = f"{prefix}_IR_{timestamp}.jpg"
            ir_full_path = self.save_dir / "IR" / ir_filename
            try:
                cv2.imwrite(str(ir_full_path), frame_ir, [cv2.IMWRITE_JPEG_QUALITY, 95])
                ir_path = str(ir_full_path)
                print(f"[ImageSaver] saved IR: {ir_path}")
            except Exception as e:
                print(f"[ImageSaver] ERROR saving IR: {e}")
        
        if frame_tv is not None:
            tv_filename = f"{prefix}_TV_{timestamp}.jpg"
            tv_full_path = self.save_dir / "TV" / tv_filename
            try:
                cv2.imwrite(str(tv_full_path), frame_tv, [cv2.IMWRITE_JPEG_QUALITY, 95])
                tv_path = str(tv_full_path)
                print(f"[ImageSaver] saved TV: {tv_path}")
            except Exception as e:
                print(f"[ImageSaver] ERROR saving TV: {e}")
        
        self.save_counter += 1
        self._cleanup_old_images("IR")
        self._cleanup_old_images("TV")
        
        return ir_path, tv_path
    
    def _cleanup_old_images(self, subdir: str = ""):
        """清理旧图像，保持数量不超过max_images"""
        target_dir = self.save_dir / subdir if subdir else self.save_dir
        
        if not target_dir.exists():
            return
        
        # 获取所有jpg文件并按时间排序
        image_files = sorted(target_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime)
        
        # 如果超过最大数量，删除最旧的
        if len(image_files) > self.max_images:
            delete_count = len(image_files) - self.max_images
            for old_file in image_files[:delete_count]:
                try:
                    old_file.unlink()
                    print(f"[ImageSaver] cleaned up: {old_file}")
                except Exception as e:
                    print(f"[ImageSaver] ERROR deleting {old_file}: {e}")
    
    def get_statistics(self) -> dict:
        """获取保存统计信息"""
        ir_count = len(list((self.save_dir / "IR").glob("*.jpg")))
        tv_count = len(list((self.save_dir / "TV").glob("*.jpg")))
        total_count = ir_count + tv_count + len(list(self.save_dir.glob("*.jpg")))
        
        return {
            "total_saved": self.save_counter,
            "ir_images": ir_count,
            "tv_images": tv_count,
            "total_files": total_count,
            "save_dir": str(self.save_dir)
        }


# 快速测试
if __name__ == "__main__":
    # 创建测试图像
    test_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    
    # 测试保存器
    saver = ImageSaver(auto_save=True, auto_interval=2)
    
    # 手动保存测试
    path = saver.save_manual(test_img, prefix="test", subdir="IR")
    print(f"Manual save: {path}")
    
    # 模拟自动保存
    import time
    for i in range(3):
        time.sleep(2.5)
        auto_path = saver.check_auto_save(test_img, prefix="auto", subdir="TV")
        if auto_path:
            print(f"Auto save: {auto_path}")
    
    # 打印统计
    stats = saver.get_statistics()
    print(f"\nStatistics: {stats}")
