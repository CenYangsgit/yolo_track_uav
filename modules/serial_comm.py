"""
Serial communication module - RS422串口通信模块
功能：支持RS422串口发送跟踪数据
"""

import serial
import struct
import time
import threading
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TrackingStatus(Enum):
    """跟踪状态枚举"""
    IDLE = 0
    DETECTING = 1
    TRACKING = 2
    LOST = 3


@dataclass
class TrackingData:
    """跟踪数据结构"""
    timestamp: float
    status: TrackingStatus
    target_x: int
    target_y: int
    target_w: int
    target_h: int
    offset_x: int
    offset_y: int
    confidence: float = 0.0
    

class SerialComm:
    """RS422串口通信类"""
    
    # 协议头
    PROTOCOL_HEADER = b'\xAA\x55'
    PROTOCOL_TAIL = b'\x0D\x0A'
    
    def __init__(self,
                 port: str = "/dev/ttyS0",
                 baudrate: int = 460800,
                 timeout: float = 0.1,
                 auto_reconnect: bool = True):
        """
        初始化串口通信
        
        Args:
            port: 串口设备路径 (如 /dev/ttyS0, /dev/ttyUSB0)
            baudrate: 波特率 (460800)
            timeout: 超时时间
            auto_reconnect: 是否自动重连
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self.send_count = 0
        self.error_count = 0
        
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        连接串口
        
        Returns:
            是否成功连接
        """
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self.connected = True
            print(f"[SerialComm] Connected to {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[SerialComm] ERROR connecting to {self.port}: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开串口连接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print(f"[SerialComm] Disconnected from {self.port}")
    
    def _try_reconnect(self) -> bool:
        """尝试重新连接"""
        if not self.auto_reconnect:
            return False
        
        print(f"[SerialComm] Attempting to reconnect...")
        self.disconnect()
        time.sleep(1)
        return self.connect()
    
    def send_tracking_data(self, data: TrackingData) -> bool:
        """
        发送跟踪数据
        
        Args:
            data: 跟踪数据对象
            
        Returns:
            是否发送成功
        """
        if not self.connected:
            if not self._try_reconnect():
                return False
        
        try:
            # 构建数据包
            packet = self._build_packet(data)
            
            with self._lock:
                self.ser.write(packet)
                self.send_count += 1
            
            return True
            
        except Exception as e:
            print(f"[SerialComm] ERROR sending data: {e}")
            self.error_count += 1
            
            if self.auto_reconnect:
                self._try_reconnect()
            
            return False
    
    def _build_packet(self, data: TrackingData) -> bytes:
        """
        构建数据包（自定义协议）
        
        协议格式：
        [Header 2B] [Timestamp 4B] [Status 1B] [X 2B] [Y 2B] [W 2B] [H 2B] 
        [OffsetX 2B] [OffsetY 2B] [Confidence 4B] [Checksum 1B] [Tail 2B]
        
        总长度：24字节
        """
        # 打包数据
        packet_data = struct.pack(
            '>HfBhhhhhhfB',
            int.from_bytes(self.PROTOCOL_HEADER, 'big'),  # Header
            data.timestamp,                                # Timestamp (4B float)
            data.status.value,                            # Status (1B)
            data.target_x,                                # X (2B)
            data.target_y,                                # Y (2B)
            data.target_w,                                # W (2B)
            data.target_h,                                # H (2B)
            data.offset_x,                                # OffsetX (2B)
            data.offset_y,                                # OffsetY (2B)
            data.confidence,                              # Confidence (4B float)
            0  # Checksum placeholder
        )
        
        # 计算校验和（简单累加）
        checksum = sum(packet_data) % 256
        
        # 重新打包，包含正确的校验和
        packet = struct.pack(
            '>HfBhhhhhhfB',
            int.from_bytes(self.PROTOCOL_HEADER, 'big'),
            data.timestamp,
            data.status.value,
            data.target_x,
            data.target_y,
            data.target_w,
            data.target_h,
            data.offset_x,
            data.offset_y,
            data.confidence,
            checksum
        ) + self.PROTOCOL_TAIL
        
        return packet
    
    def send_simple(self, 
                   x: int, y: int, 
                   offset_x: int, offset_y: int,
                   status: TrackingStatus = TrackingStatus.TRACKING) -> bool:
        """
        发送简化的跟踪数据
        
        Args:
            x, y: 目标坐标
            offset_x, offset_y: 偏移量
            status: 跟踪状态
            
        Returns:
            是否发送成功
        """
        data = TrackingData(
            timestamp=time.time(),
            status=status,
            target_x=x,
            target_y=y,
            target_w=0,
            target_h=0,
            offset_x=offset_x,
            offset_y=offset_y,
            confidence=0.95
        )
        return self.send_tracking_data(data)
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "connected": self.connected,
            "port": self.port,
            "baudrate": self.baudrate,
            "send_count": self.send_count,
            "error_count": self.error_count,
            "success_rate": (self.send_count - self.error_count) / max(1, self.send_count) * 100
        }


# 测试代码
if __name__ == "__main__":
    # 注意：需要实际的串口设备才能测试
    # 这里使用虚拟串口或loopback进行测试
    
    print("Serial Communication Module Test")
    print("=" * 50)
    
    # 创建串口对象（可能需要修改端口名）
    comm = SerialComm(port="/dev/ttyS0", baudrate=460800, auto_reconnect=True)
    
    # 尝试连接
    if comm.connect():
        print("✓ Connection successful")
        
        # 发送测试数据
        for i in range(5):
            data = TrackingData(
                timestamp=time.time(),
                status=TrackingStatus.TRACKING,
                target_x=100 + i * 10,
                target_y=200 + i * 5,
                target_w=50,
                target_h=50,
                offset_x=-20 + i,
                offset_y=15 - i,
                confidence=0.95
            )
            
            success = comm.send_tracking_data(data)
            print(f"Send #{i+1}: {'✓ OK' if success else '✗ FAIL'}")
            time.sleep(0.1)
        
        # 打印统计
        stats = comm.get_statistics()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 断开连接
        comm.disconnect()
        print("✓ Disconnected")
        
    else:
        print("✗ Connection failed")
        print("Note: Make sure the serial port exists and has proper permissions")
        print("      You may need to run: sudo chmod 666 /dev/ttyS0")
