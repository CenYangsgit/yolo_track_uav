此文件汇总了项目各脚本的常用执行命令。
**说明：**
1. 所有命令默认启用了 `--pattern` 以使用模拟源（无需真实相机）。如需连接真实相机，请去掉 `--pattern` 参数。
2. 默认启用了 `--headless` (无图形界面) 和 `--quiet` (简化日志)，适合 SSH 或后台运行。如需看画面，请去掉 `--headless`。

---

## 1. 实时流检测与追踪 (detect_track_camera.py)

这是核心脚本，支持模拟源或真实相机流的检测与追踪。

### 1.1 红外模式 (IR Only)

python3 detect_track_camera.py --mode ir --pattern --headless --max_frames 200 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --save_csv_ir ./results/ir_csv/ir1.csv   （不推流）

python3 detect_track_camera.py --mode ir --pattern --headless --max_frames 0 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --save_csv_ir ./results/ir_csv/ir1.csv --rtsp_ir /ir --rtsp_encoder x264 --rtsp_bitrate 4096  (推流，软件编码x264)

python3 detect_track_camera.py --mode ir --pattern --headless --max_frames 0 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --save_csv_ir ./results/ir_csv/ir1.csv --rtsp_ir /ir --rtsp_encoder mpph264enc --rtsp_bitrate 4906 (推流，硬件编码mpph264enc)

---

### 1.2 可见光模式 (TV Only)
python3 detect_track_camera.py --mode tv --pattern --headless --max_frames 200 --quiet --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 10 --save_csv_tv ./results/tv_csv/tv1.csv

python3 detect_track_camera.py --mode tv --pattern --headless --max_frames 0 --quiet --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 10 --save_csv_tv ./results/tv_csv/tv1.csv --rtsp_tv /tv --rtsp_encoder x264 --rtsp_bitrate 4096 (推流，软件编码x264)

python3 detect_track_camera.py --mode tv --pattern --headless --max_frames 0 --quiet --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 10 --save_csv_tv ./results/tv_csv/tv1.csv --rtsp_tv /tv --rtsp_encoder mpph264enc --rtsp_bitrate 4096 (推流，硬件编码mpph264enc)
---


### 1.3 双光模式 (IR + TV)
python3 detect_track_camera.py --mode both --pattern --headless --max_frames 200 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 15 --save_csv_ir ./results/ir_csv/ir2.csv --save_csv_tv ./results/tv_csv/tv2.csv  （不推流）

python3 detect_track_camera.py --mode both --pattern --headless --max_frames 0 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 15 --save_csv_ir ./results/ir_csv/ir2.csv --save_csv_tv ./results/tv_csv/tv2.csv --rtsp_ir /ir --rtsp_tv /tv --rtsp_encoder x264 --rtsp_bitrate 4096 (推流，软件编码x264)

python3 detect_track_camera.py --mode both --pattern --headless --max_frames 0 --quiet --ir_model ./weights/IR.rknn --ir_img_size 640x640 --frame_skip_ir 10 --tv_model ./weights/TV.rknn --tv_img_size 1088x1088 --frame_skip_tv 15 --save_csv_ir ./results/ir_csv/ir3.csv --save_csv_tv ./results/tv_csv/tv3.csv --rtsp_ir /ir --rtsp_tv /tv --rtsp_encoder  mpph264enc --rtsp_bitrate 4096 (推流，硬件编码mpph264enc)
---

## 2. 离线视频/图片文件检测 (detect_track.py)

用于跑通已经录制好的视频或图片集，不涉及摄像头调用。

### 2.1 单视频文件追踪
**红外视频 (IR):**
```bash
python3 detect_track.py --model_path ./weights/IR.rknn \
  --video_path ./datasets/video/IR/IR.mp4 \
  --img_size 640x640 --save_video --result_tag IR_Demo_Video
```

**可见光视频 (TV):**
```bash
python3 detect_track.py --model_path ./weights/TV.rknn \
  --video_path ./datasets/video/TV/TV.mp4 \
  --img_size 1088x1088 --save_video --result_tag TV_Demo_Video
```

### 2.2 图片文件夹检测
```bash
python3 detect_track.py --model_path ./weights/IR.rknn \
  --img_folder ./datasets/images/IR \
  --img_size 640x640 --img_save --result_tag IR_Image_Test
```

---

## 3. 简单的相机测试 (camera_test.py)

仅用于测试摄像头是否能打开、模拟源是否正常，不加载模型。

**测试红外模拟源:**
```bash
python3 camera_test.py --mode ir --pattern --max-frames 100
```

**测试双路模拟源:**
```bash
python3 camera_test.py --mode both --pattern --max-frames 100

```

## 4.precision_eval.py
python3 ./tools/precision_eval.py --csv_ir ./results/ir_csv/ir1.csv --csv_tv ./results/tv_csv/tv1.csv --metric l2 --pctl 95