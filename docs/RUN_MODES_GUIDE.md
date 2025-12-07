# 运行模式与推荐参数组合

本指南整理了 `detect_track_camera.py` 的常用运行模式与推荐参数组合，方便根据不同需求在**高精度 / 高帧率 / 低功耗 / 调试**之间快速切换。

> 说明：所有命令均假设当前工作目录为 `~/yolo_track_uav`，并使用 `python3` 运行。根据实际情况可将 `python3` 换成 `python`，将串口设备号替换为真实端口。

---

## 1. 高精度模式（优先检测质量和稳定性）

- 特点：
  - 每帧都进行检测（不跳帧）；
  - 开启卡尔曼滤波平滑偏移；
  - 开启 OSD（若接显示器时）更易观察。
- 适合：验收测试、算法精度评估、复杂背景场景。

```bash
python3 detect_track_camera.py \
  --mode both \
  --headless \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_ir 0 \
  --frame_skip_tv 0 \
  --save_csv ./results/detect_track/both_high_quality.csv \
  --quiet
```

如需在无显示器环境下启用 Web 手动框选，在命令末尾追加：

```bash
  --manual_select_web
```

---

## 2. 高帧率模式（优先 FPS）

- 特点：
  - 通过 `frame_skip` 减少检测频率，多数帧仅使用 Tracker；
  - 适当关闭 OSD，降低绘制开销；
  - 在满足协议帧率的前提下争取更高 FPS。
- 适合：目标运动较快，对实时性要求较高的场景。

```bash
python3 detect_track_camera.py \
  --mode both \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_ir 3 \
  --frame_skip_tv 2 \
  --save_csv ./results/detect_track/both_high_fps.csv \
  --quiet
```

> 提示：如果板子负载仍偏高，可进一步增大 `frame_skip_ir` / `frame_skip_tv`，或临时降低 `--rtsp_bitrate`。

---

## 3. 低功耗 / 长时间运行模式

- 特点：
  - 大幅提高跳帧系数，减少推理次数；
  - 关闭 OSD，降低绘制与编码压力；
  - 可降低 RTSP 码率，减少网络与编码负载；
  - 可选地关闭 CSV，避免长时间运行占用过多磁盘空间。
- 适合：长时间在线监视、对本地记录要求不高的场景。

```bash
python3 detect_track_camera.py \
  --mode both \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 2048 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_ir 5 \
  --frame_skip_tv 4 \
  --save_csv "" \
  --quiet
```

> 说明：若不需要 CSV，可直接省略 `--save_csv` 参数，或将路径改为 `/tmp/xxx.csv` 等临时目录。

---

## 4. 调试 + Web 手动框选模式（Pattern 源）

- 特点：
  - 使用 `--pattern` 无需真实相机，在实验环境快速跑通全链路；
  - `--manual_select_web` 启用 Web 手动框选，方便在浏览器中交互调试；
  - 适合开发阶段验证算法、RTSP、串口与 Web 控制是否协同工作。

```bash
python3 detect_track_camera.py \
  --mode both \
  --pattern \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/pts/3 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_ir 0 \
  --frame_skip_tv 0 \
  --save_csv ./results/detect_track/both_web_debug.csv \
  --quiet \
  --manual_select_web
```

- 在 PC 浏览器中访问：

```text
http://<开发板IP>:5000/manual
```

- 左侧 IR、右侧 TV 均可通过鼠标拖拽框选，终端中会打印：

```text
[IR] manual bbox applied via web: (x, y, w, h)
[TV] manual bbox applied via web: (x, y, w, h)
```

---

## 5. 单路调试模板（IR / TV 任意切换）

适合只接入一类相机或需要单路排查时使用。

### 5.1 单路 IR 示例

```bash
python3 detect_track_camera.py \
  --mode ir \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_ir 2 \
  --quiet \
  --manual_select_web
```

### 5.2 单路 TV 示例

```bash
python3 detect_track_camera.py \
  --mode tv \
  --headless \
  --no_osd \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --frame_skip_tv 2 \
  --quiet \
  --manual_select_web
```

> 提示：在仿真环境下，可将串口部分替换为 `--serial_port /dev/pts/3`（配合 socat 虚拟串口），其余参数保持不变。

---

通过以上几套预制命令，你可以根据现场需求快速在**高精度 / 高帧率 / 低功耗 / 调试**之间切换，同时保持协议要求的多模态处理、RTSP 输出和串口通信能力。