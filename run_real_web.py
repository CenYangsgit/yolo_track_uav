#!/usr/bin/env python3
"""一键启动脚本（实机相机 + RTSP + Web 手动框选 + 串口输出）

场景：
- 已接入真实 IR / TV 摄像头（通过 /dev/video*）（先确认这一步）

- 需要同时：
  - 运行检测 + 跟踪
  - 通过 RTSP 推流到 VLC 等客户端
  - 通过 Web 页面在 PC 上进行手动框选修正
  - 通过 RS422 串口输出偏移等协议数据

使用前准备：
1. 在开发板上进入工程目录：

   cd ~/yolo_track_uav

2. 确认真实串口设备号（例如 /dev/ttyS3 或 /dev/ttyUSB0）：
   可用 "dmesg | grep tty"、"ls /dev/ttyS* /dev/ttyUSB*" 辅助确认。

3. 将本脚本顶部的 SERIAL_PORT 修改为实际串口设备路径。

4. 如果需要在 PC 上抓取串口数据做调试，可外接 USB-RS422 转接头，
   然后在 PC 端用串口调试工具监听即可，本脚本无需修改。

5. 运行：

   python3 run_real_web.py

之后可以：
- 在 VLC 中打开 rtsp://<板子IP>:8554/ir 与 /tv
- 在浏览器中打开 http://<板子IP>:5000/manual 进行手动框选
"""

import os
import sys
import subprocess

# ==== 根据板子实际串口设备修改这里 =======================================
# 实机 RS422 串口设备路径，例如 /dev/ttyS3 或 /dev/ttyUSB0
SERIAL_PORT = "/dev/ttyS3"  # TODO: 根据实际硬件修改
# 串口波特率（协议要求为 460800，可按协议调整）
SERIAL_BAUD = 460800
# ========================================================================


def main() -> int:
    # 切换到脚本所在目录（工程根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("============================================================")
    print(" 一键启动：实机相机 + RTSP + Web 手动框选 + 串口输出")
    print("------------------------------------------------------------")
    print(f" 串口设备: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(" 请确认：")
    print("  1) 真实 IR / TV 摄像头已经正确接入，对应 /dev/video* 设备可用；")
    print("  2) 串口设备路径 SERIAL_PORT 已根据实际硬件修改；")
    print("  3) 若需在 PC 上查看串口数据，请将 RS422 接到 PC，并在 PC 端开启监听。")
    print("============================================================\n")

    cmd = [
        sys.executable or "python3",      # 使用当前解释器
        "detect_track_camera.py",
        "--mode", "both",               # 同时运行 IR + TV
        # 实机相机模式下不加 --pattern，默认从 /dev/video* 读取
        "--headless",                    # 推荐在无显示器 / 仅 SSH 环境下使用
        # 如需在本地 HDMI 显示器上看到窗口，可注释掉上一行
        # "--no_osd",                    # 如需先关闭 OSD，可取消注释
        # 可根据需要单独调整 IR/TV 模型与输入尺寸
        # "--ir_model", "./weights/IR.rknn",
        # "--tv_model", "./weights/TV.rknn",
        # "--ir_img_size", "640x640",
        # "--tv_img_size", "1088x1088",
        # 若需要跳帧检测，可取消注释并设置间隔：
        # "--frame_skip_ir", "0",       # IR 每隔 N 帧检测
        # "--frame_skip_tv", "0",       # TV 每隔 N 帧检测
        "--rtsp_ir", "/ir",            # IR RTSP mount point
        "--rtsp_tv", "/tv",            # TV RTSP mount point
        "--rtsp_encoder", "mpph264enc", # RK3588 硬件 H.264 编码
        "--rtsp_bitrate", "4096",       # 码率 4096 kbps
        "--serial_port", SERIAL_PORT,    # 串口发送端（真实 RS422 口）
        "--serial_baud", str(SERIAL_BAUD),  # 串口波特率
        "--use_kalman",                  # 启用卡尔曼滤波
        # CSV 统计：合并 + 分路，可按需要修改路径
        "--save_csv", "./results/detect_track/both_real.csv",      # 合并偏移统计（IR+TV）
        # "--save_csv_ir", "./results/detect_track/ir_real.csv",   # 仅 IR
        # "--save_csv_tv", "./results/detect_track/tv_real.csv",   # 仅 TV
        # 若希望自动在跑满一定帧数后退出，可设置：
        # "--max_frames", "0",           # 0 表示不限制
        "--quiet",                        # 简要日志
        "--manual_select_web",           # 启用 Web 手动框选
    ]

    print("将执行命令:\n  " + " ".join(cmd) + "\n")

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[run_real_web] 捕获到 Ctrl+C，准备退出...")
        return 0
    except Exception as exc:  # pragma: no cover - 防御性输出
        print(f"[run_real_web] 启动失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
