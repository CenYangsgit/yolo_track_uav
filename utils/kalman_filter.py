"""
Kalman filter for target tracking prediction
卡尔曼滤波器 - 用于目标轨迹预测和平滑
"""

import numpy as np
from typing import Optional, Tuple


class KalmanFilter:
    """
    卡尔曼滤波器用于2D目标跟踪
    状态向量: [x, y, vx, vy] (位置和速度)
    """
    
    def __init__(self, dt: float = 1.0, process_noise: float = 1.0, measurement_noise: float = 1.0):
        """
        初始化卡尔曼滤波器
        
        Args:
            dt: 时间步长（帧间隔）
            process_noise: 过程噪声（预测不确定性）
            measurement_noise: 测量噪声（观测不确定性）
        """
        self.dt = dt
        
        # 状态向量 [x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float32)
        
        # 状态转移矩阵 F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵 H (只能观测到位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差 Q
        q = process_noise
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q, 0],
            [0, 0, 0, q]
        ], dtype=np.float32)
        
        # 测量噪声协方差 R
        r = measurement_noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float32)
        
        # 误差协方差矩阵 P
        self.P = np.eye(4, dtype=np.float32) * 1000
        
        # 初始化标志
        self.initialized = False
        
        # 预测历史（用于可视化）
        self.history = []
    
    def initialize(self, x: float, y: float):
        """
        初始化滤波器状态
        
        Args:
            x: 初始x坐标
            y: 初始y坐标
        """
        self.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.initialized = True
        self.history = [(x, y)]
    
    def predict(self) -> Tuple[float, float]:
        """
        预测下一步状态
        
        Returns:
            (predicted_x, predicted_y) 预测的坐标
        """
        if not self.initialized:
            return 0.0, 0.0
        
        # 预测状态: x' = F * x
        self.x = np.dot(self.F, self.x)
        
        # 预测误差协方差: P' = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        pred_x = float(self.x[0, 0])
        pred_y = float(self.x[1, 0])
        
        return pred_x, pred_y
    
    def update(self, measured_x: float, measured_y: float) -> Tuple[float, float]:
        """
        使用测量值更新状态
        
        Args:
            measured_x: 测量的x坐标
            measured_y: 测量的y坐标
            
        Returns:
            (updated_x, updated_y) 更新后的坐标
        """
        if not self.initialized:
            self.initialize(measured_x, measured_y)
            return measured_x, measured_y
        
        # 测量向量
        z = np.array([[measured_x], [measured_y]], dtype=np.float32)
        
        # 创新（残差）: y = z - H * x
        y = z - np.dot(self.H, self.x)
        
        # 创新协方差: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # 卡尔曼增益: K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新状态: x = x + K * y
        self.x = self.x + np.dot(K, y)
        
        # 更新误差协方差: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        updated_x = float(self.x[0, 0])
        updated_y = float(self.x[1, 0])
        
        # 记录历史
        self.history.append((updated_x, updated_y))
        if len(self.history) > 100:  # 保留最近100个点
            self.history.pop(0)
        
        return updated_x, updated_y
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        获取当前估计的速度
        
        Returns:
            (vx, vy) 速度向量
        """
        if not self.initialized:
            return 0.0, 0.0
        
        return float(self.x[2, 0]), float(self.x[3, 0])
    
    def predict_future(self, steps: int = 5) -> list:
        """
        预测未来N步的位置
        
        Args:
            steps: 预测步数
            
        Returns:
            [(x1, y1), (x2, y2), ...] 预测位置列表
        """
        if not self.initialized:
            return []
        
        # 保存当前状态
        x_backup = self.x.copy()
        P_backup = self.P.copy()
        
        predictions = []
        for _ in range(steps):
            # 执行预测
            self.x = np.dot(self.F, self.x)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            
            pred_x = float(self.x[0, 0])
            pred_y = float(self.x[1, 0])
            predictions.append((pred_x, pred_y))
        
        # 恢复状态
        self.x = x_backup
        self.P = P_backup
        
        return predictions
    
    def reset(self):
        """重置滤波器"""
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1000
        self.initialized = False
        self.history = []


class BboxKalmanFilter:
    """
    用于跟踪边界框的卡尔曼滤波器
    状态向量: [cx, cy, w, h, vcx, vcy, vw, vh]
    (中心坐标、宽高、速度)
    """
    
    def __init__(self, dt: float = 1.0):
        """
        初始化边界框卡尔曼滤波器
        
        Args:
            dt: 时间步长
        """
        self.dt = dt
        self.x = np.zeros((8, 1), dtype=np.float32)
        
        # 状态转移矩阵
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt
        
        # 观测矩阵
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        
        # 噪声矩阵
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0
        self.P = np.eye(8, dtype=np.float32) * 1000
        
        self.initialized = False
    
    def initialize(self, bbox: Tuple[int, int, int, int]):
        """
        初始化边界框
        
        Args:
            bbox: (x, y, w, h)
        """
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        
        self.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 1000
        self.initialized = True
    
    def predict(self) -> Tuple[int, int, int, int]:
        """预测下一个边界框"""
        if not self.initialized:
            return 0, 0, 0, 0
        
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        cx, cy, w, h = self.x[0:4, 0]
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return x, y, int(w), int(h)
    
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """更新边界框"""
        if not self.initialized:
            self.initialize(bbox)
            return bbox
        
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        
        y_residual = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y_residual)
        I = np.eye(8, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        cx, cy, w, h = self.x[0:4, 0]
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        return x, y, int(w), int(h)
    
    def reset(self):
        """重置"""
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 1000
        self.initialized = False


# 测试代码
if __name__ == "__main__":
    # 测试位置卡尔曼滤波器
    kf = KalmanFilter(dt=1.0, process_noise=1.0, measurement_noise=5.0)
    
    # 模拟带噪声的轨迹
    true_trajectory = [(i * 10, i * 5) for i in range(20)]
    noisy_measurements = [(x + np.random.randn() * 5, y + np.random.randn() * 5) 
                         for x, y in true_trajectory]
    
    filtered_trajectory = []
    for mx, my in noisy_measurements:
        kf.predict()
        fx, fy = kf.update(mx, my)
        filtered_trajectory.append((fx, fy))
    
    print("True vs Noisy vs Filtered (last 5 points):")
    for i in range(-5, 0):
        print(f"True: {true_trajectory[i]}, "
              f"Noisy: ({noisy_measurements[i][0]:.1f}, {noisy_measurements[i][1]:.1f}), "
              f"Filtered: ({filtered_trajectory[i][0]:.1f}, {filtered_trajectory[i][1]:.1f})")
    
    # 测试未来预测
    future = kf.predict_future(steps=5)
    print(f"\nFuture predictions (next 5 steps): {future}")
