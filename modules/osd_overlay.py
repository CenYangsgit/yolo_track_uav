"""
OSD overlay module - OSD叠加显示模块
功能：绘制跟踪框、十字准线、坐标信息、状态信息等
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from datetime import datetime


class OSDOverlay:
    """OSD叠加渲染器"""
    
    # 颜色定义 (BGR)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_CYAN = (255, 255, 0)
    
    def __init__(self, 
                 font_scale: float = 0.6,
                 font_thickness: int = 2,
                 language: str = "zh",
                 crosshair_color: Tuple[int, int, int] = COLOR_CYAN,
                 box_color_tracking: Tuple[int, int, int] = COLOR_GREEN,
                 box_color_detecting: Tuple[int, int, int] = COLOR_YELLOW,
                 box_color_lost: Tuple[int, int, int] = COLOR_RED,
                 text_color: Tuple[int, int, int] = COLOR_WHITE):
        """
        初始化OSD渲染器
        
        Args:
            font_scale: 字体大小缩放
            font_thickness: 字体线条粗细
            language: 语言 ("zh" 中文 或 "en" 英文)
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.language = language
        # 颜色配置（可通过构造参数自定义）
        self.crosshair_color = crosshair_color
        self.box_color_tracking = box_color_tracking
        self.box_color_detecting = box_color_detecting
        self.box_color_lost = box_color_lost
        self.text_color = text_color
        
        # 文本映射
        self.text_map = {
            "zh": {
                "tracking": "跟踪中",
                "detecting": "检测中",
                "lost": "目标丢失",
                "target": "目标",
                "offset": "偏移",
                "size": "尺寸",
                "fps": "帧率",
                "frame": "帧数"
            },
            "en": {
                "tracking": "Tracking",
                "detecting": "Detecting",
                "lost": "Lost",
                "target": "Target",
                "offset": "Offset",
                "size": "Size",
                "fps": "FPS",
                "frame": "Frame"
            }
        }
    
    def get_text(self, key: str) -> str:
        """获取本地化文本"""
        return self.text_map.get(self.language, self.text_map["en"]).get(key, key)
    
    def draw_tracking_box(self, 
                         frame, 
                         bbox: Optional[Tuple[int, int, int, int]],
                         color=COLOR_GREEN,
                         thickness: int = 2) -> None:
        """
        绘制跟踪框
        
        Args:
            frame: 图像帧
            bbox: 边界框 (x, y, w, h)
            color: 颜色
            thickness: 线条粗细
        """
        if bbox is None:
            return
        
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # 绘制角点标记
        corner_len = min(20, w // 5, h // 5)
        # 左上角
        cv2.line(frame, (x, y), (x + corner_len, y), color, thickness + 1)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thickness + 1)
        # 右上角
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thickness + 1)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thickness + 1)
        # 左下角
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thickness + 1)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thickness + 1)
        # 右下角
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 1)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 1)
    
    def draw_crosshair(self, 
                      frame,
                      center: Optional[Tuple[int, int]] = None,
                      size: int = 30,
                      color=None,
                      thickness: int = 2) -> None:
        """
        绘制十字准线
        
        Args:
            frame: 图像帧
            center: 中心点坐标，None则使用图像中心
            size: 十字线长度
            color: 颜色
            thickness: 线条粗细
        """
        h, w = frame.shape[:2]
        if color is None:
            color = self.crosshair_color
        cx, cy = center if center else (w // 2, h // 2)
        
        # 绘制十字线
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)
        
        # 绘制中心圆
        cv2.circle(frame, (cx, cy), 5, color, -1)
    
    def draw_target_info(self,
                        frame,
                        bbox: Optional[Tuple[int, int, int, int]],
                        offset: Optional[Tuple[int, int]] = None,
                        status: str = "detecting") -> None:
        """
        在跟踪框旁边绘制目标信息
        
        Args:
            frame: 图像帧
            bbox: 边界框 (x, y, w, h)
            offset: 偏移量 (offset_x, offset_y)
            status: 状态 ("tracking", "detecting", "lost")
        """
        if bbox is None:
            return
        
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        
        # 状态文本颜色
        status_color = {
            "tracking": self.box_color_tracking,
            "detecting": self.box_color_detecting,
            "lost": self.box_color_lost
        }.get(status, self.text_color)
        
        # 准备文本内容
        status_text = self.get_text(status)
        pos_text = f"({cx}, {cy})"
        size_text = f"{self.get_text('size')}: {w}x{h}"
        
        # 计算文本位置（在框的上方）
        text_y = max(y - 10, 30)
        
        # 绘制状态
        self._draw_text_with_bg(frame, status_text, (x, text_y), status_color)
        
        # 绘制坐标
        self._draw_text_with_bg(frame, pos_text, (x, text_y + 25), self.text_color)
        
        # 绘制尺寸
        self._draw_text_with_bg(frame, size_text, (x, text_y + 50), self.text_color)
        
        # 如果有偏移量，绘制偏移信息
        if offset is not None:
            offx, offy = offset
            offset_text = f"{self.get_text('offset')}: ({offx:+d}, {offy:+d})"
            self._draw_text_with_bg(frame, offset_text, (x, text_y + 75), self.crosshair_color)
    
    def draw_stats_panel(self,
                        frame,
                        fps: float = 0.0,
                        frame_idx: int = 0,
                        mode: str = "IR",
                        position: str = "top-left") -> None:
        """
        绘制统计信息面板
        
        Args:
            frame: 图像帧
            fps: 帧率
            frame_idx: 帧数
            mode: 模式 ("IR", "TV", "BOTH")
            position: 位置 ("top-left", "top-right", "bottom-left", "bottom-right")
        """
        h, w = frame.shape[:2]
        
        # 准备文本
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fps_text = f"{self.get_text('fps')}: {fps:.1f}"
        frame_text = f"{self.get_text('frame')}: {frame_idx}"
        mode_text = f"Mode: {mode}"
        
        # 确定位置
        if position == "top-left":
            base_x, base_y = 10, 25
        elif position == "top-right":
            base_x, base_y = w - 250, 25
        elif position == "bottom-left":
            base_x, base_y = 10, h - 100
        else:  # bottom-right
            base_x, base_y = w - 250, h - 100
        
        # 绘制背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (base_x - 5, base_y - 20), 
                     (base_x + 240, base_y + 80), 
                     self.COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 绘制文本
        cv2.putText(frame, timestamp, (base_x, base_y), 
                   self.font, self.font_scale * 0.7, self.COLOR_WHITE, 1)
        cv2.putText(frame, fps_text, (base_x, base_y + 25), 
                   self.font, self.font_scale, self.COLOR_GREEN, self.font_thickness)
        cv2.putText(frame, frame_text, (base_x, base_y + 50), 
               self.font, self.font_scale, self.text_color, self.font_thickness)
        cv2.putText(frame, mode_text, (base_x, base_y + 75), 
                   self.font, self.font_scale, self.COLOR_CYAN, self.font_thickness)
    
    def draw_complete_osd(self,
                         frame,
                         bbox: Optional[Tuple[int, int, int, int]] = None,
                         offset: Optional[Tuple[int, int]] = None,
                         status: str = "detecting",
                         fps: float = 0.0,
                         frame_idx: int = 0,
                         mode: str = "IR",
                         show_crosshair: bool = True) -> None:
        """
        绘制完整的OSD（一站式调用）
        
        Args:
            frame: 图像帧
            bbox: 边界框
            offset: 偏移量
            status: 状态
            fps: 帧率
            frame_idx: 帧数
            mode: 模式
            show_crosshair: 是否显示十字准线
        """
        # 绘制统计面板
        self.draw_stats_panel(frame, fps, frame_idx, mode, "top-left")
        
        # 绘制十字准线
        if show_crosshair:
            self.draw_crosshair(frame)
        
        # 如果有目标，绘制跟踪框和信息
        if bbox is not None:
            status_color = {
                "tracking": self.COLOR_GREEN,
                "detecting": self.COLOR_YELLOW,
                "lost": self.COLOR_RED
            }.get(status, self.COLOR_WHITE)
            
            self.draw_tracking_box(frame, bbox, color=status_color, thickness=2)
            self.draw_target_info(frame, bbox, offset, status)
    
    def _draw_text_with_bg(self, 
                          frame, 
                          text: str, 
                          pos: Tuple[int, int],
                          color=COLOR_WHITE,
                          bg_color=COLOR_BLACK) -> None:
        """绘制带背景的文本"""
        x, y = pos
        
        # 获取文本尺寸
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, self.font_scale * 0.8, self.font_thickness - 1
        )
        
        # 绘制背景
        cv2.rectangle(frame, 
                     (x - 2, y - text_h - 2), 
                     (x + text_w + 2, y + baseline + 2),
                     bg_color, -1)
        
        # 绘制文本
        cv2.putText(frame, text, (x, y), 
                   self.font, self.font_scale * 0.8, color, self.font_thickness - 1)


# 测试代码
if __name__ == "__main__":
    import numpy as np
    
    # 创建测试图像
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)  # 深灰色背景
    
    # 创建OSD渲染器
    osd = OSDOverlay(language="zh")
    
    # 模拟跟踪框
    bbox = (500, 300, 200, 150)
    offset = (-50, 30)
    
    # 绘制完整OSD
    osd.draw_complete_osd(
        test_img,
        bbox=bbox,
        offset=offset,
        status="tracking",
        fps=25.5,
        frame_idx=1234,
        mode="IR",
        show_crosshair=True
    )
    
    # 显示结果
    cv2.imshow("OSD Test", test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
