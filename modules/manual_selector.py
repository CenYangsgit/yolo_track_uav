"""
Manual target selector - 手动目标选择模块
功能：支持鼠标框选目标进行跟踪
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable


class ManualSelector:
    """手动目标选择器 - 支持鼠标拖拽选择"""
    
    def __init__(self, window_name: str = "Manual Selection"):
        """
        初始化手动选择器
        
        Args:
            window_name: 窗口名称
        """
        self.window_name = window_name
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.current_frame = None
        self.selected_bbox = None
        
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始选择
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 拖拽中
            if self.selecting:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成选择
            self.selecting = False
            self.end_point = (x, y)
            
            # 计算边界框
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # 确保坐标正确（左上到右下）
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1, x2)
                y_max = max(y1, y2)
                
                w = x_max - x_min
                h = y_max - y_min
                
                # 只有当框选区域足够大时才认为有效
                if w > 10 and h > 10:
                    self.selected_bbox = (x_min, y_min, w, h)
    
    def select_target(self, 
                     frame, 
                     instruction: str = "Drag to select target, press SPACE to confirm, ESC to cancel",
                     timeout: int = 30000) -> Optional[Tuple[int, int, int, int]]:
        """
        在图像上选择目标
        
        Args:
            frame: 输入图像
            instruction: 指导文本
            timeout: 超时时间（毫秒）
            
        Returns:
            选中的边界框 (x, y, w, h)，取消则返回None
        """
        self.current_frame = frame.copy()
        self.selected_bbox = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # 显示指导文本
        instruction_zh = "拖拽选择目标，SPACE确认，ESC取消"
        
        start_time = cv2.getTickCount()
        
        while True:
            display_frame = self.current_frame.copy()
            
            # 绘制当前选择框
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制指导文本
            self._draw_instruction(display_frame, instruction_zh)
            
            cv2.imshow(self.window_name, display_frame)
            
            # 检查按键
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC - 取消
                self.selected_bbox = None
                break
            elif key == 32:  # SPACE - 确认
                if self.selected_bbox:
                    break
            
            # 检查超时
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            if elapsed > timeout:
                print("[ManualSelector] Selection timeout")
                self.selected_bbox = None
                break
        
        cv2.destroyWindow(self.window_name)
        return self.selected_bbox
    
    def _draw_instruction(self, frame, text: str):
        """绘制指导文本"""
        h, w = frame.shape[:2]
        
        # 背景矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 60), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (20, h - 25), font, 0.7, (255, 255, 255), 2)


class InteractiveSelector:
    """交互式选择器 - 支持多种选择方式"""
    
    def __init__(self):
        self.manual_selector = ManualSelector()
    
    def select_by_click(self, 
                       frame,
                       click_expand: int = 50) -> Optional[Tuple[int, int, int, int]]:
        """
        单击选择（自动扩展为边界框）
        
        Args:
            frame: 输入图像
            click_expand: 点击后扩展的像素数
            
        Returns:
            边界框 (x, y, w, h)
        """
        selected_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point[0] = (x, y)
        
        window_name = "Click to Select"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        instruction = "Click on target center, press SPACE to confirm, ESC to cancel"
        instruction_zh = "点击目标中心，SPACE确认，ESC取消"
        
        while True:
            display = frame.copy()
            
            if selected_point[0]:
                cx, cy = selected_point[0]
                # 绘制十字
                cv2.drawMarker(display, (cx, cy), (0, 255, 0), 
                             cv2.MARKER_CROSS, 30, 2)
                # 绘制预览框
                x = max(0, cx - click_expand)
                y = max(0, cy - click_expand)
                w = click_expand * 2
                h = click_expand * 2
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制指导
            h_frame, w_frame = frame.shape[:2]
            cv2.putText(display, instruction_zh, (20, h_frame - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                selected_point[0] = None
                break
            elif key == 32 and selected_point[0]:  # SPACE
                break
        
        cv2.destroyWindow(window_name)
        
        if selected_point[0]:
            cx, cy = selected_point[0]
            x = max(0, cx - click_expand)
            y = max(0, cy - click_expand)
            w = click_expand * 2
            h = click_expand * 2
            return (x, y, w, h)
        
        return None
    
    def select_by_drag(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """
        拖拽选择
        
        Args:
            frame: 输入图像
            
        Returns:
            边界框 (x, y, w, h)
        """
        return self.manual_selector.select_target(frame)


# 测试代码
if __name__ == "__main__":
    # 创建测试图像
    test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 添加一些"目标"
    cv2.rectangle(test_img, (300, 200), (500, 400), (0, 255, 0), -1)
    cv2.rectangle(test_img, (700, 300), (900, 500), (255, 0, 0), -1)
    
    # 测试手动选择
    selector = InteractiveSelector()
    
    print("Testing drag selection...")
    bbox = selector.select_by_drag(test_img)
    if bbox:
        print(f"Selected bbox: {bbox}")
        x, y, w, h = bbox
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow("Result", test_img)
        cv2.waitKey(2000)
    else:
        print("Selection cancelled")
    
    # 测试点击选择
    print("\nTesting click selection...")
    test_img2 = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cv2.circle(test_img2, (640, 360), 50, (255, 255, 0), -1)
    
    bbox2 = selector.select_by_click(test_img2, click_expand=60)
    if bbox2:
        print(f"Selected bbox: {bbox2}")
        x, y, w, h = bbox2
        cv2.rectangle(test_img2, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imshow("Result", test_img2)
        cv2.waitKey(2000)
    else:
        print("Selection cancelled")
    
    cv2.destroyAllWindows()
