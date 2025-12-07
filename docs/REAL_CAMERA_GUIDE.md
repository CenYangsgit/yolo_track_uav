# 实机测试指南（接入真实相机与 RS422）

本指南说明在 **RK3588 开发板接入真实红外/可见光相机与 RS422 串口** 的条件下，如何部署与运行 `detect_track_camera.py`，完成协议要求的功能验证。

根据实际使用场景，分为两类：

- **模式 A：无本地显示器（推荐最终部署形态）**  —— 只依赖网口 + 串口，通过 RTSP 和上位机串口工具完成观察与记录；
- **模式 B：接显示器 + 键鼠（实验室/厂内联调与功能演示）** —— 在板子本地屏幕上显示 OSD 画面并支持手动选目标，同时仍通过 RTSP/串口将数据送往上位机。

---

## 1. 硬件连接与前提

1. **相机接入**
   - 红外相机 → MIPI CSI 接口，对应内核中的一个 `/dev/videoN` 设备；
   - 可见光相机 → 另一路 MIPI CSI 接口，对应另一个 `/dev/videoM` 设备。

2. **串口接入**
   - RS422 差分收发器连接到 RK3588 板上的 UART（例如 `/dev/ttyS4`，具体以实际为准）。
   - 波特率需配置为 **460800 8N1**（协议要求）。

3. **网络环境**
   - 开发板与 PC 处于同一网段；
   - PC 上安装 VLC，用于接收 RTSP 码流；
   - 开发板上已部署本工程代码与模型权重。

4. **依赖安装**（首次执行即可）：

```bash
cd ~/yolo_track_uav
pip3 install -r requirements.txt
```

---

## 2. 查找视频与串口设备（建议先用 `tools/v4l2_probe.py`）

1. **推荐：先用脚本探测相机能力**

在开发板上进入工程根目录：

```bash
cd ~/yolo_track_uav
python3 tools/v4l2_probe.py
```

- 脚本会内部调用 `v4l2-ctl --list-devices` 和 `--list-formats-ext`，并把结果保存到 `configs/cameras.yaml`（纯文本，供人工查看/记录）；
- 你可以在终端输出中查看每个 `/dev/videoN` 的分辨率、像素格式等支持情况，方便确认哪一路是 IR、哪一路是 TV；
- 若后续想长期保存当前板子的摄像头配置，可手工在 `configs/cameras.yaml` 里做注释标记（例如：`IR -> /dev/video22`，`TV -> /dev/video11`）。

2. **手工列出视频设备（可选，等价于脚本里的命令）**：

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

- 观察各 `/dev/videoN` 与对应描述（MIPI CSI 摄像头一般会有清晰标识）。
- 确认：
  - 红外相机 → 例如 `/dev/video22`
  - 可见光相机 → 例如 `/dev/video11`

3. **列出串口设备**：

```bash
ls -l /dev/ttyS* /dev/ttyUSB* 2>/dev/null
```

- 结合硬件手册，确认 RS422 接在如 `/dev/ttyS4` 上。

3. **必要时修改代码中的设备号**

在 `detect_track_camera.py` 顶部可根据实际修改：

```python
IR_DEVICE_ID = 22   # /dev/video22
TV_DEVICE_ID = 11   # /dev/video11
```

> 如果你的实际编号不同，只需改为对应的数字（不改时使用默认 22/11）。

---

## 3. 单路相机功能检查

### 3.1 红外通道（IR）

在开发板上执行：

```bash
cd ~/yolo_track_uav

python3 detect_track_camera.py \
  --mode ir \
  --headless \
  --no_osd \
  --rtsp_ir /ir \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --quiet
```

- 观察终端输出是否有：
  - `[IR] opened /dev/video22 as ... size=640x512, FPS≈50.00`（尺寸/FPS 以协议为目标值）
  - `[rtsp] Stream ready at rtsp://<IP>:8554/ir ...`

### 3.2 可见光通道（TV）

```bash
cd ~/yolo_track_uav

python3 detect_track_camera.py \
  --mode tv \
  --headless \
  --no_osd \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --quiet
```

- 期望输出：
  - `[TV] opened /dev/video11 as ... size=1920x1080, FPS≈25.00`
  - `[rtsp] Stream ready at rtsp://<IP>:8554/tv ...`

### 3.3 使用 VLC 验证画面

在 PC 上（假设板子 IP 为 `192.168.137.10`）：

- 打开红外流：`rtsp://192.168.137.10:8554/ir`
- 打开可见光流：`rtsp://192.168.137.10:8554/tv`

确保画面清晰、延迟和帧率满足需求。

---

## 4. 模式 A：无本地显示器（HEADLESS，推荐最终部署）

在该模式下：

- 板子 **不接显示器**，你通过 SSH（如 MobaXterm）登录板子；
- `detect_track_camera.py` 使用 `--headless`，不弹出本地窗口；
- 图像通过 RTSP 推到上位机，用 VLC 观看；
- 串口数据通过 RS422 → USB 转换器发送到上位机，在 PC 上使用串口助手接收。

### 4.1 双光联合运行（不带串口）

在开发板上运行：

```bash
cd ~/yolo_track_uav

python3 detect_track_camera.py \
  --mode both \
  --headless \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --quiet
```

- 这一步确认：
  - IR/TV 两路能同步工作；
  - 双路 RTSP 码流稳定输出；
  - 终端 `frames=... FPS≈...` 显示的帧率达到协议要求（红外约 50fps，可见光约 25fps）。

---

### 4.2 接入真实 RS422 串口（HEADLESS 模式）

1. **给予串口访问权限**（以 `/dev/ttyS4` 为例）：

```bash
sudo chmod 666 /dev/ttyS4
ls -l /dev/ttyS4
```

2. **运行主程序（带串口）**：

```bash
cd ~/yolo_track_uav

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
  --save_csv ./results/detect_track/both.csv \
  --quiet
```

- 正常启动时，终端应打印：

```text
[serial] connected to /dev/ttyS4 @ 460800
```

3. **上位机/示波器验证协议**

- 使用上位机串口工具（通过 RS422 → USB 转换）监听同一串口线路；
- 检查帧格式是否符合协议：
  - 头 0xAA55，尾 0x0D0A，总长度 26 字节；
  - 内含时间戳、状态、bbox、偏移、置信度等字段；
- 通过移动目标/切换场景，观察状态 `TRACKING/LOST` 与偏移值变化是否合理。

> **推荐的上位机串口工具（Windows）：**
> 
> - MobaXterm 自带 Serial Session（你已经在用，方便集成）；
> - 或常用的 "sscom" 串口助手（适合查看十六进制数据和时间戳）。

在 HEADLESS 部署形态下，功能覆盖：

- 双光实时检测 + 跟踪；
- RTSP/H.264 网络输出；
- RS422 串口输出目标状态、位置与偏移；
- CSV 记录历史数据。此模式基本对应最终在无人机/车载平台上的实际运行形态。

---

## 5. 模式 B：有本地显示器（OSD + 手动选目标）

该模式用于**实验室/工厂联调和功能演示**：

- 开发板接 HDMI（或其他显示接口）显示器，以及 USB 键盘/鼠标；
- 直接在板子本地终端运行程序，不再使用 `--headless`；
- 画面与 OSD 在板子显示器上实时显示，支持鼠标框选目标；
- 同时仍可通过网口推 RTSP 到上位机，通过 RS422 向上位机发送串口数据。

若开发板接有 HDMI 显示器与 USB 键盘/鼠标，可在**非 headless 模式**下运行：

```bash
cd ~/yolo_track_uav

python3 detect_track_camera.py \
  --mode both \
  --rtsp_ir /ir \
  --rtsp_tv /tv \
  --rtsp_encoder mpph264enc \
  --rtsp_bitrate 4096 \
  --serial_port /dev/ttyS4 \
  --serial_baud 460800 \
  --use_kalman \
  --save_csv ./results/detect_track/both.csv \
  --manual_select \
  --quiet

---

## 6.2 Web 手动框选（实机环境）

当板子仅通过 SSH 访问、无本地显示器时，可以通过浏览器进行手动框选，不影响 RTSP 推流和串口输出。

1. 在开发板上启动主程序（示例为双路 IR+TV）：

```bash
cd ~/yolo_track_uav

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
  --quiet \
  --manual_select_web
```

2. 在 PC 浏览器中访问（假设开发板 IP 为 `192.168.137.10`）：

```text
http://192.168.137.10:5000/manual
```

3. 页面说明：

- 左侧为 IR 红外画面，可用鼠标拖拽矩形，松开后即向设备发送手动框选 → 终端打印：

  ```text
  [IR] manual bbox applied via web: (x, y, w, h)
  ```

- 右侧为 TV 可见光画面，同样支持拖拽，终端打印：

  ```text
  [TV] manual bbox applied via web: (x, y, w, h)
  ```

4. 适用于：

- 无本地显示器，仅通过 RTSP + 浏览器进行观测与人工干预；
- 也可搭配真实相机使用，无需 `--pattern`。
```

### 6.1 OSD 功能验证

- 屏幕上会显示：
  - 十字准线；
  - 当前帧率 FPS、帧号 Frame；
  - 模式标识（IR/TV）；
  - 目标框、目标中心坐标、宽高、偏移量等信息。

### 6.2 手动选目标操作

- 在主画面窗口：
  - 按 **`m`**：进入手动框选模式；
  - 在弹出的选择窗口，用鼠标拖拽出一个矩形框，按 **空格** 确认；
  - 按 **ESC** 取消选择；
  - 成功选择后，跟踪器和卡尔曼滤波会以该框为新的目标继续跟踪。

### 6.3 退出程序

- 在主窗口按 **`q`**，或在终端按 `Ctrl+C` 均可停止运行。

---
---

## 8. 建议的测试顺序小结

1. 按第 2～4 章：在 **模式 A（Headless）** 下依次完成：单路 IR → 单路 TV → 双路 Both（先不带串口，再加上串口）。
2. 在 Headless 模式下，用 VLC 和上位机串口工具，确认 RTSP+串口链路长期稳定。
3. 若需要验收 OSD 与手动选目标，再使用 **模式 B（有显示器）**：接显示器+键鼠，取消 `--headless` 并加上 `--manual_select`，在本地屏幕上验证 OSD/交互效果，同时继续用上位机串口工具与 VLC 观测。
4. 通过 CSV 和串口数据，对比实景中的像素误差、目标尺寸范围、复杂背景表现，完成整体性能评估。
   - CSV 文件：记录 IR/TV bbox 与偏移随时间变化。
5. **图像输出能力**：RTSP + H.264，满足“网络输出、RTSP 协议、H.264 压缩编码”的要求。
6. **视频参数**：
   - 通过 `IR_WIDTH/IR_HEIGHT/IR_FPS` 与 `TV_WIDTH/TV_HEIGHT/TV_FPS` 设置 640×512@50fps 与 1920×1080@25fps；
   - 可用终端 `FPS≈` 输出及 VLC/测量工具确认是否达到指标。
7. **通信接口**：
   - RS422 串口，波特率 460800，协议已按需求实现；
   - 1000M 网口 + IP 通过串口配置的功能，可在此基础上后续扩展（当前版本未实现 IP 配置指令）。

---

## 8. 建议的测试顺序小结

1. 单路 IR → 单路 TV → 双路 Both（均不带串口），逐级验证视频链路与 RTSP。
2. 加入串口输出（`--serial_port=/dev/ttyS*`），用上位机验证协议内容。
3. 接显示器后打开 OSD 与手动选目标，验证交互与显示效果。
4. 通过 CSV 和串口数据，对比实景中的像素误差、目标尺寸范围、复杂背景表现，完成整体性能评估。
