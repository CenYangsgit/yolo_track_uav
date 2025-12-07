# 模拟链路测试指南（无真实相机 / 无真实 RS422）

本指南说明在 **只有 RK3588 开发板 + SSH（无显示器、无相机、无 RS422 硬件）** 的条件下，如何完整跑通：

> 模型加载 → 双路视频模拟 → 检测 + 跟踪 → 串口协议 → RTSP 推流 → VLC 预览

---

## 1. 环境与目录准备

1. 登陆开发板，进入工程目录：

```bash
cd ~/yolo_track_uav
```

2. 确认 Python 依赖（只需首次执行）：

```bash
pip3 install -r requirements.txt
sudo apt-get install -y socat
```

3. 确保以下关键文件/目录存在：

- `detect_track_camera.py`
- `yolov8_ir.py`, `yolov8_tv.py`
- `weights/IR.rknn`, `weights/TV.rknn`
- `modules/serial_comm.py`, `modules/osd_overlay.py`, `modules/manual_selector.py`
- `utils/kalman_filter.py`
- `tools/rtsp_push_appsrc.py`, `tools/serial_monitor.py`

> 这些文件在你当前仓库中已经存在，本指南只说明如何使用。

---

## 2. 启动虚拟串口（socat）

1. 在 **终端 A** 中执行：

```bash
cd ~/yolo_track_uav
sudo socat -d -d pty,raw,echo=0 pty,raw,echo=0
```

2. 终端会输出类似信息：

```text
N PTY is /dev/pts/3
N PTY is /dev/pts/4
N starting data transfer loop with FDs [...]
```

- 记住这两个设备号（如 `/dev/pts/3` 与 `/dev/pts/4`）。
- **保持终端 A 常开，不要 Ctrl+C**，否则虚拟串口会消失。

---

## 3. 设置虚拟串口权限

1. 打开 **终端 B**，执行（按你实际的 pts 号修改）：

```bash
cd ~/yolo_track_uav
sudo chmod 666 /dev/pts/4 /dev/pts/5
ls -l /dev/pts/4 /dev/pts/5
```

2. 确认权限类似：

```text
crw-rw-rw- 1 root tty ... /dev/pts/3
crw-rw-rw- 1 root tty ... /dev/pts/4
```

---

## 4. 启动串口监听工具

在 **终端 B**（或新开 **终端 C**）中运行：

```bash
cd ~/yolo_track_uav
python3 tools/serial_monitor.py /dev/pts/5 460800
```

- `/dev/pts/4` 为监听端；稍后主程序会向 `/dev/pts/3` 发送数据。
- 成功后界面显示：

```text
串口监听启动
端口: /dev/pts/4
波特率: 460800
...
等待数据...(Ctrl+C 停止)
```

保持该监听程序运行，用于验证协议数据。

---

## 5. 启动主程序（pattern + headless + RTSP + 串口）

在 **终端 C** 中执行（假设发送端是 `/dev/pts/3`）：

```bash
cd ~/yolo_track_uav

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
  --save_csv ./results/detect_track/both.csv \
  --quiet
```

### 5.1 关键参数说明

- `--mode both`：同时运行 IR/TV 两路。
- `--pattern`：使用 `videotestsrc` 生成测试图像，不需要真实相机。
- `--headless`：关闭 OpenCV 窗口显示，适合仅 SSH 的环境。
- `--no_osd`：不开启 OSD 叠加，先简化为裸画面。
- `--rtsp_ir /ir`, `--rtsp_tv /tv`：RTSP 服务挂载点，分别为 `/ir` 与 `/tv`。
- `--rtsp_encoder mpph264enc`：使用 RK3588 硬件 H.264 编码。
- `--serial_port /dev/pts/3`：虚拟串口发送端。
- `--use_kalman`：对目标中心使用卡尔曼滤波，平滑偏移。
- `--save_csv`：记录 IR/TV bbox 与偏移到 CSV 文件。
- `--quiet`：在终端持续打印帧数与 FPS。

### 5.2 正常启动时应看到的日志

- 串口连接：

```text
[serial] connected to /dev/pts/3 @ 460800
```

- 视频 pipeline：

```text
[IR] opened videotestsrc as NV12, size=640x512, FPS≈50.00
[TV] opened videotestsrc as NV12, size=1920x1080, FPS≈25.00
```

- RTSP：

```text
[rtsp] Stream ready at rtsp://<板子IP>:8554/ir ...
[rtsp] Stream ready at rtsp://<板子IP>:8554/tv ...
```

- 帧数统计：

```text
[run] frames=  120 FPS≈ 24.7
```

若看到以上信息，说明模拟链路已经处于正常运行状态。

---

## 6. 在 PC 上使用 VLC 预览 RTSP 流

1. 假设开发板 IP 为 `192.168.137.10`，先在 Windows 命令行确认网络连通：

```cmd
ping 192.168.137.10
```

2. 在 PC 上打开 VLC：

- 选择「媒体」→「打开网络串流」。
- 分别输入：
  - 红外流：`rtsp://192.168.137.10:8554/ir`
  - 可见光流：`rtsp://192.168.137.10:8554/tv`

3. 看到彩色测试画面（移动的圆球）即说明：

- GStreamer → RK3588 硬件编码 → RTSP Server → 网络 → VLC 整条链路正常。

---

## 7. 验证串口协议输出

继续观察 **终端 B**（`serial_monitor.py`）：

- 应看到连续的数据包，例如：

```text
数据包 #1
时间戳 : 1764842880.000
状态   : LOST / TRACKING
目标   : X=..., Y=..., W=..., H=...
偏移   : X=..., Y=...
置信度 : 0.xx
校验和 : 0xYY
```

- 无异常报错时，串口链路已验证通过。

如果长时间无数据：

- 检查 socat 是否仍在运行；
- 确认主程序是否打印 `[serial] connected ...`；
- 确认 `--serial_port` 与监听端两头的设备号对应正确（一个用 `/dev/pts/3`，一个用 `/dev/pts/4`）。

---

## 8. 启用 OSD 叠加（可选）

在确认基础链路稳定后，可以在命令中移除 `--no_osd`：

```bash
python3 detect_track_camera.py \
  --mode both \
  --pattern \
  --headless \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/pts/4 \
  --serial_baud 460800 \
  --use_kalman \
  --save_csv ./results/detect_track/both.csv \
  --quiet
```

此时通过 VLC 看到的画面会叠加：

- 帧率、帧号、模式（IR/TV）；
- 十字准线；
- 当前目标框与坐标、尺寸、偏移等信息。

---

## 9. Web 手动框选调试（HEADLESS + pattern）

在完全无显示器、仅通过 SSH 的环境下，也可以使用 Web 浏览器在 PC 端手动框选目标，对跟踪进行修正。该功能不影响原有 RTSP 推流和串口输出，仅作为额外交互通道。

### 9.1 启动带 Web 控制的主程序

在 **终端 C**（主程序终端）中运行：

```bash
cd ~/yolo_track_uav

python3 detect_track_camera.py \
  --mode both \
  --pattern \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder x264 \
  --rtsp_bitrate 4096 \
  --serial_port /dev/pts/5 \
  --serial_baud 460800 \
  --use_kalman \
  --save_csv ./results/detect_track/both.csv \
  --quiet \
  --manual_select_web
```

关键新增参数：

- `--manual_select_web`：
  - 启用 Web 手动框选功能；
  - 程序会在端口 `5000` 上启动一个 Flask Web 服务；
  - 不改变 RTSP/H.264 推流逻辑。

启动成功后，终端应多出类似日志：

```text
[web] manual control enabled: http://<board-ip>:5000/manual
... "GET /manual HTTP/1.1" 200 -
... "GET /video_ir HTTP/1.1" 200 -
... "GET /video_tv HTTP/1.1" 200 -
```

### 9.2 在 PC 浏览器中打开 Web 控制页面

假设开发板 IP 为 `192.168.137.10`，在 Windows 浏览器地址栏输入：

```text
http://192.168.137.10:5000/manual
```

页面上可以看到：

- 左侧：IR 红外预览（仅查看，不在此页面上框选）；
- 右侧：TV 可见光预览，并叠加一个透明 Canvas 用于鼠标操作。

### 9.3 在 TV 画面上进行手动框选

1. 将鼠标移动到 TV 画面区域；
2. 按住左键，从目标左上角向右下拖动，划出一个矩形；
3. 松开鼠标后，浏览器会自动向开发板发送一个 `POST /manual_tv` 请求，携带 `(x, y, w, h)`；
4. 若发送成功，页面下方状态栏会显示：

```text
TV: 手动框选已发送，bbox=[x, y, w, h]
```

与此同时，在开发板主程序终端会看到类似日志：

```text
[TV] manual bbox applied via web: (x, y, w, h)
```

说明：

- 程序已使用你划定的矩形重新初始化 TV 跟踪器；
- 后续偏移量/串口输出/CSV 记录都会基于新框计算；
- IR 通道也支持 Web 手动选框，只需在页面左侧 IR 画面实现对应交互后，通过 `POST /manual_ir` 即可（当前页面默认只对 TV 开启拖拽）。

### 9.4 单路模式下的 Web 手动框选

若只跑单路 IR 或单路 TV，`--manual_select_web` 同样有效：

- 仅 IR：

  ```bash
  python3 detect_track_camera.py \
    --mode ir \
    --pattern \
    --headless \
    --no_osd \
    --rtsp_ir /ir \
    --rtsp_encoder mpph264enc \
    --rtsp_bitrate 4096 \
    --use_kalman \
    --quiet \
    --manual_select_web
  ```

- 仅 TV：

  ```bash
  python3 detect_track_camera.py \
    --mode tv \
    --pattern \
    --headless \
    --no_osd \
    --rtsp_tv /tv \
    --rtsp_encoder mpph264enc \
    --rtsp_bitrate 4096 \
    --use_kalman \
    --quiet \
    --manual_select_web
  ```

此时 Web 页面中只会显示对应一路的预览画面，手动框选仍可用。RTSP 推流和串口输出逻辑保持不变。

---

## 10. 结束程序

- 结束串口监听：在运行 `serial_monitor.py` 的终端按 `Ctrl+C`；
- 停止主程序：在运行 `detect_track_camera.py` 的终端按 `Ctrl+C`；
- 最后停止 socat：在终端 A 按 `Ctrl+C`。

---

## 11. 与协议要求的覆盖关系（模拟场景）

在上述模拟链路下，已经验证：

- 同时处理红外 / 可见光（IR+TV 双路 pattern 源）。
- 自动检测 + 跟踪流程（YOLOv8 + OpenCV Tracker）。
- 图像输出能力（RTSP + H.264，已被 VLC 打开）。
- 串口输出能力（虚拟串口 + 自定义协议 + 监听工具解析）。

与真实系统的差异主要在于：

- 图像源为 `videotestsrc`，而非 MIPI CSI 摄像头；
- 串口物理介质为虚拟 TTY，而非真实 RS422；
- 网口 IP 通过串口配置的逻辑尚未在本仓库中实现。

其他流程（检测、跟踪、偏移计算、串口打包、RTSP 推流）与实机运行保持一致，可视为验收前的软件自检方案。
