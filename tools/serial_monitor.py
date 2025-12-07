#!/usr/bin/env python3
"""
串口监听工具 - Serial Port Monitor
用于测试和调试串口通信

使用方法:
    python3 serial_monitor.py /dev/ttyS0 460800
    python3 serial_monitor.py /dev/pts/3 460800
"""

import sys
import serial
import time
import struct
from datetime import datetime


def parse_tracking_packet(data):
    """
    解析跟踪数据包
    
    协议格式（参考 modules/serial_comm.py）:
    [Header 2B] [Timestamp 4B] [Status 1B] [X 2B] [Y 2B] [W 2B] [H 2B]
    [OffsetX 2B] [OffsetY 2B] [Confidence 4B] [Checksum 1B] [Tail 2B]
    数据段 + 校验 + 尾部总长度: 26字节
    """
    if len(data) < 26:
        return None
    
    try:
        # 解包数据
        unpacked = struct.unpack('>HfBhhhhhhfBH', data[:26])
        
        header = unpacked[0]
        timestamp = unpacked[1]
        status = unpacked[2]
        target_x = unpacked[3]
        target_y = unpacked[4]
        target_w = unpacked[5]
        target_h = unpacked[6]
        offset_x = unpacked[7]
        offset_y = unpacked[8]
        confidence = unpacked[9]
        checksum = unpacked[10]
        tail = unpacked[11]
        
        # 验证协议头和尾
        if header != 0xAA55:
            return None
        if tail != 0x0D0A:
            return None
        
        # 状态映射
        status_map = {
            0: "IDLE",
            1: "DETECTING",
            2: "TRACKING",
            3: "LOST"
        }
        
        return {
            'timestamp': timestamp,
            'status': status_map.get(status, f"UNKNOWN({status})"),
            'target': (target_x, target_y, target_w, target_h),
            'offset': (offset_x, offset_y),
            'confidence': confidence,
            'checksum': checksum
        }
    except Exception as e:
        print(f"解析错误: {e}")
        return None


def monitor_serial(port, baudrate, parse=True):
    """监听串口数据"""
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        print("=" * 70)
        print(f"📡 串口监听启动")
        print(f"   端口: {port}")
        print(f"   波特率: {baudrate}")
        print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print("\n等待数据... (Ctrl+C 停止)\n")
        
        packet_count = 0
        buffer = b''
        
        while True:
            if ser.in_waiting > 0:
                # 读取可用数据
                data = ser.read(ser.in_waiting)
                buffer += data
                
                # 如果启用解析，尝试解析完整数据包
                if parse and len(buffer) >= 26:
                    # 查找协议头 0xAA55
                    header_pos = buffer.find(b'\xAA\x55')
                    
                    if header_pos != -1:
                        # 提取一个完整数据包（26字节）
                        if len(buffer) >= header_pos + 26:
                            packet = buffer[header_pos:header_pos + 26]
                            buffer = buffer[header_pos + 26:]  # 移除已处理的包
                            
                            # 解析数据包
                            parsed = parse_tracking_packet(packet)
                            if parsed:
                                packet_count += 1
                                print(f"📦 数据包 #{packet_count}")
                                print(f"   时间戳: {parsed['timestamp']:.3f}")
                                print(f"   状态: {parsed['status']}")
                                print(f"   目标: X={parsed['target'][0]}, Y={parsed['target'][1]}, "
                                      f"W={parsed['target'][2]}, H={parsed['target'][3]}")
                                print(f"   偏移: X={parsed['offset'][0]:+d}, Y={parsed['offset'][1]:+d}")
                                print(f"   置信度: {parsed['confidence']:.2f}")
                                print(f"   校验和: 0x{parsed['checksum']:02X}")
                                print("-" * 70)
                            else:
                                print(f"⚠️  无效数据包: {packet.hex()}")
                    else:
                        # 清理缓冲区（防止溢出）
                        if len(buffer) > 1024:
                            buffer = buffer[-512:]
                else:
                    # 不解析，直接打印十六进制
                    print(f"📨 接收 ({len(data)} bytes): {data.hex()}")
                    print(f"   ASCII: {data.decode('ascii', errors='ignore')}")
                    print("-" * 70)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  停止监听")
        print(f"📊 统计: 共接收 {packet_count} 个数据包")
        
    except serial.SerialException as e:
        print(f"\n❌ 串口错误: {e}")
        print("\n💡 常见问题:")
        print("   1. 检查设备是否存在: ls -l /dev/ttyS* 或 ls -l /dev/pts/*")
        print("   2. 检查权限: sudo chmod 666 <设备路径>")
        print("   3. 确认设备未被占用: fuser -k <设备路径>")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("✓ 串口已关闭")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 serial_monitor.py <串口设备> [波特率] [--raw]")
        print("\n示例:")
        print("  python3 serial_monitor.py /dev/ttyS0 460800")
        print("  python3 serial_monitor.py /dev/pts/3 460800")
        print("  python3 serial_monitor.py /dev/ttyS0 460800 --raw  # 不解析，仅显示原始数据")
        print("\n说明:")
        print("  串口设备: 串口设备路径（如 /dev/ttyS0, /dev/pts/3）")
        print("  波特率: 默认 460800")
        print("  --raw: 显示原始十六进制数据，不解析协议")
        sys.exit(1)
    
    port = sys.argv[1]
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 460800
    parse = '--raw' not in sys.argv
    
    monitor_serial(port, baudrate, parse)


if __name__ == "__main__":
    main()
