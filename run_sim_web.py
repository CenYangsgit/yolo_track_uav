#!/usr/bin/env python3
"""一键启动脚本：

功能：
- 启动 detect_track_camera.py
- 使用 pattern 模式模拟 IR/TV 双路
- 启用 RTSP 推流（ir/tv）
- 启用 Web 手动框选（http://<板子IP>:5000/manual）
- 启用串口输出（配合 socat + serial_monitor.py）

使用前准备：
1. 在开发板上进入工程目录：

   cd ~/yolo_track_uav

2. 在终端 A 启动 socat，得到一对虚拟串口，例如：

   sudo socat -d -d pty,raw,echo=0 pty,raw,echo=0
   # 假设输出：/dev/pts/4 与 /dev/pts/5

3. 在终端 B 赋予权限：

   sudo chmod 666 /dev/pts/4 /dev/pts/5

4. 在终端 B（或 C）启动监听（这里假设监听端是 /dev/pts/5）：

   python3 tools/serial_monitor.py /dev/pts/5 460800

5. 根据实际情况在本脚本中设置 SERIAL_PORT / MONITOR_PORT，
   保证：主程序发送端 == SERIAL_PORT，serial_monitor 监听端 == MONITOR_PORT。

之后只需执行：

   python3 run_sim_web.py

即可启动完整链路。
"""

import os
import sys
import subprocess

# ==== 根据你的实际 socat 输出修改这里 =====================================
# 主程序用于发送数据的串口（例如 /dev/pts/4）
SERIAL_PORT = "/dev/pts/4"

# 串口监听工具使用的端口，仅用于提示（不影响主程序运行逻辑）
MONITOR_PORT = "/dev/pts/5"
# ========================================================================


def main() -> int:
    # 切换到脚本所在目录（工程根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    cmd = [
        sys.executable or "python3",      # 使用当前解释器
        "detect_track_camera.py",
        "--mode", "both",               # 同时运行 IR + TV
        "--pattern",                     # 使用 videotestsrc 模拟图像
        "--headless",                    # 不弹出 OpenCV 窗口
        "--no_osd",                      # 先关闭 OSD，确保链路简单稳定
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
        "--serial_port", SERIAL_PORT,    # 串口发送端
        "--serial_baud", "460800",      # 串口波特率
        "--use_kalman",                  # 启用卡尔曼滤波
      # CSV 统计：合并 + 分路，可按需要修改路径
      "--save_csv", "./results/detect_track/both.csv",      # 合并偏移统计（IR+TV）
      # "--save_csv_ir", "./results/detect_track/ir_only.csv",  # 仅 IR
      # "--save_csv_tv", "./results/detect_track/tv_only.csv",  # 仅 TV
      # 若希望自动在跑满一定帧数后退出，可设置：
      # "--max_frames", "0",           # 0 表示不限制
        "--quiet",                        # 简要日志
        "--manual_select_web",           # 启用 Web 手动框选
    ]

    print("将执行命令:\n  " + " ".join(cmd) + "\n")

    try:
        # 直接继承当前终端的输入输出，方便观察日志
        result = subprocess.run(cmd)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[run_sim_web] 捕获到 Ctrl+C，准备退出...")
        return 0
    except Exception as exc:  # pragma: no cover - 仅防御性输出
        print(f"[run_sim_web] 启动失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
