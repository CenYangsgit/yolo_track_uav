"""
detect_track_camera.py 增强版（支持单独 CSV 与分别跳帧）

新增：
- --save_csv_ir、--save_csv_tv：分别保存 IR/TV 的统计
- --frame_skip_ir、--frame_skip_tv：分别控制 IR/TV 的检测间隔
- 保留 --save_csv（合并写入），三者可并存
"""

import os
import sys
import time
import argparse
import os.path as osp
from types import SimpleNamespace

import cv2
import numpy as np
from py_utils.coco_utils import COCO_test_helper
from modules import SerialComm, TrackingData, TrackingStatus
from modules.osd_overlay import OSDOverlay
from modules.manual_selector import InteractiveSelector
from utils.kalman_filter import KalmanFilter, BboxKalmanFilter
from web.manual_control import set_shared_state, run_app_in_thread

# 模块回退逻辑
y8_ir = None
y8_tv = None
y8_default = None
try:
    import yolov8_ir as y8_ir
except Exception as e:
    print("[warn] import yolov8_ir failed:", e)
try:
    import yolov8_tv as y8_tv
except Exception as e:
    print("[warn] import yolov8_tv failed:", e)

IR_DEVICE_ID = 22
IR_WIDTH, IR_HEIGHT, IR_FPS = 640, 512, 50

TV_DEVICE_ID = 11
TV_WIDTH, TV_HEIGHT, TV_FPS = 1920, 1080, 25

PREFERRED_FORMATS = ["NV12", "UYVY","YUY2", "YUYV", "NV16"]


# Web 交互共享状态（通过 dict 包一层，便于多线程共享引用）
latest_ir_frame_ref = {"frame": None}
latest_tv_frame_ref = {"frame": None}
manual_bbox_ir_ref = {"bbox": None}
manual_bbox_tv_ref = {"bbox": None}


def build_pipeline(device_id, width, height, fps, pix_fmt, use_pattern=False):
    if use_pattern:
        return ("videotestsrc is-live=true pattern=ball ! "
                f"video/x-raw,format={pix_fmt},width={width},height={height},framerate={fps}/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=1")
    return (f"v4l2src device=/dev/video{device_id} ! "
            f"video/x-raw,format={pix_fmt},width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1")


def open_camera(device_id, width, height, fps, use_pattern=False, name="cam"):
    src_name = "videotestsrc" if use_pattern else f"/dev/video{device_id}"
    for fmt in PREFERRED_FORMATS:
        pipeline = build_pipeline(device_id, width, height, fps, fmt, use_pattern)
        print(f"[{name}] try format={fmt} src={src_name}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rf = cap.get(cv2.CAP_PROP_FPS) or 0.0
            print(f"[{name}] opened {src_name} as {fmt}, size={rw}x{rh}, FPS≈{rf:.2f}")
            return cap
        cap.release()
    print(f"[{name}] ERROR: cannot open {src_name} with {PREFERRED_FORMATS}")
    if not use_pattern:
        print(f"[{name}] hint: v4l2-ctl --list-formats-ext -d {src_name}")
    return None


def parse_size(s: str):
    s = s.lower().replace('×', 'x').replace(',', 'x')
    a = s.split('x')
    if len(a) == 1:
        v = int(a[0]); return v, v
    return int(a[0]), int(a[1])


def _select_y8_module(model_path: str):
    name = (osp.basename(model_path) if model_path else "").lower()
    if 'ir' in name and y8_ir is not None:
        return y8_ir, 'ir'
    if 'tv' in name and y8_tv is not None:
        return y8_tv, 'tv'
    if y8_ir is not None:
        return y8_ir, 'ir'
    if y8_tv is not None:
        return y8_tv, 'tv'
    raise RuntimeError("No yolov8 module available.")


def setup_y8(module, model_path, branches, img_size, quiet,
             obj_thresh=0.25, nms_thresh=0.45):
    module.OBJ_THRESH = float(obj_thresh)
    module.NMS_THRESH = float(nms_thresh)
    module.BRANCHES = int(branches)
    if not hasattr(module, "INPUT_LAYOUT"):
        module.INPUT_LAYOUT = "nchw"

    if img_size:
        w, h = parse_size(img_size)
        module.IMG_SIZE = (int(w), int(h))
    module.QUIET = bool(quiet)

    ns = SimpleNamespace(model_path=model_path, target='rk3588', device_id=None, quiet=quiet)
    print(f"[model] loading {model_path} with module={getattr(module, '__name__', 'y8')} ...")
    model, platform = module.setup_model(ns)
    print(f"[model] loaded (platform={platform})")

    if not img_size and platform == 'rknn':
        try:
            ops = model.rknn.get_input_ops()
            if ops and 'shape' in ops[0]:
                shp = ops[0]['shape']
                if len(shp) == 4:
                    if shp[1] in (1, 3):
                        module.INPUT_LAYOUT = 'nchw'
                        module.IMG_SIZE = (int(shp[3]), int(shp[2]))
                    elif shp[-1] in (1, 3):
                        module.INPUT_LAYOUT = 'nhwc'
                        module.IMG_SIZE = (int(shp[2]), int(shp[1]))
            print(f"[model] inferred IMG_SIZE={module.IMG_SIZE}, INPUT_LAYOUT={module.INPUT_LAYOUT}")
        except Exception as e:
            print("[model] WARNING: infer input shape failed:", e)

    return model, platform


def setup_model_with_fallback(model_path, branches, img_size, quiet):
    mod, mod_name = _select_y8_module(model_path)
    try:
        model, plat = setup_y8(mod, model_path, branches, img_size, quiet)
        return mod, model, plat
    except Exception as e:
        print(f"[model] ERROR with module={mod_name}: {e}")
        if y8_default and mod is not y8_default:
            print("[model] fallback to yolov8 (default) ...")
            model, plat = setup_y8(y8_default, model_path, branches, img_size, quiet)
            return y8_default, model, plat
        raise


def infer_once(module, model, platform, frame_bgr, helper):
    W, H = module.IMG_SIZE
    img = helper.letter_box(frame_bgr.copy(), new_shape=(H, W), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if platform in ['pytorch', 'onnx']:
        inp = img.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
    else:
        if getattr(module, "INPUT_LAYOUT", "nchw") == 'nchw':
            inp = img.transpose(2, 0, 1)[None, ...].astype(np.uint8)
        else:
            inp = np.expand_dims(img, 0).astype(np.uint8)

    try:
        outputs = model.run([inp])
    except Exception as e:
        print(f"[infer] ERROR: model.run failed: {e}; shape={getattr(inp,'shape',None)}; layout={getattr(module,'INPUT_LAYOUT',None)}")
        return None, None, None

    if not outputs:
        return None, None, None

    boxes, classes, scores = module.post_process(outputs)
    if boxes is None:
        return None, None, None
    real_boxes = helper.get_real_box(boxes)
    return real_boxes, classes, scores


def pick_primary_target(boxes, scores):
    if boxes is None or scores is None or len(scores) == 0:
        return None
    idx = int(np.argmax(scores))
    x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (x1, y1, w, h)


def create_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    raise RuntimeError("OpenCV lacks CSRT/KCF trackers. Install opencv-contrib-python.")


def center_offset(center, bbox):
    if bbox is None:
        return None, None
    cx, cy = center
    x, y, w, h = bbox
    tx, ty = x + w // 2, y + h // 2
    return tx - cx, ty - cy


def build_arg_parser():
    ap = argparse.ArgumentParser("dual camera detect+track (IR + TV, RK3588)")
    ap.add_argument('--mode', choices=['ir', 'tv', 'both'], default='both',
                    help='选择运行模式：ir 仅红外；tv 仅可见光；both 双路')
    ap.add_argument('--pattern', action='store_true', help='使用 videotestsrc 进行模拟（无摄像头时）')
    ap.add_argument('--headless', action='store_true', help='不弹出窗口显示（适合 SSH/无图形环境）')
    ap.add_argument('--max_frames', type=int, default=0, help='最多运行多少帧后自动退出（0 表示不限制）')
    ap.add_argument('--quiet', action='store_true', help='安静模式，打印简要进度')

    # 合并 CSV（包含两路）
    ap.add_argument('--save_csv', type=str, default='./results/both.csv',
                    help='合并偏移统计 CSV 文件路径（包含 IR 与 TV）')

    # 分路 CSV（可选，分别保存）
    ap.add_argument('--save_csv_ir', type=str, default=None,
                    help='仅保存 IR 的偏移与状态统计 CSV 路径')
    ap.add_argument('--save_csv_tv', type=str, default=None,
                    help='仅保存 TV 的偏移与状态统计 CSV 路径')

    # 模型与输入尺寸
    ap.add_argument('--ir_model', type=str, default='./weights/IR.rknn')
    ap.add_argument('--tv_model', type=str, default='./weights/TV.rknn')
    ap.add_argument('--ir_img_size', type=str, default='640x640')
    ap.add_argument('--tv_img_size', type=str, default='1088x1088')

    # 分别跳帧
    ap.add_argument('--frame_skip_ir', type=int, default=0,
                    help='IR 的检测帧间隔（>1 表示每隔 N 帧检测，其余帧仅跟踪）')
    ap.add_argument('--frame_skip_tv', type=int, default=0,
                    help='TV 的检测帧间隔')

    # RTSP 推流（可选）
    ap.add_argument('--rtsp_ir', type=str, default=None, help='红外 RTSP mount point，例如 /ir')
    ap.add_argument('--rtsp_tv', type=str, default=None, help='可见光 RTSP mount point，例如 /tv')
    ap.add_argument('--rtsp_encoder', type=str, choices=['x264', 'mpph264enc'], default='x264',
                    help='H.264 编码器类型（软件 x264 或硬件 mpph264enc）')
    ap.add_argument('--rtsp_bitrate', type=int, default=4096, help='H.264 码率（kbps）')

    # 串口输出（协议要求：RS422 460800bps）
    ap.add_argument('--serial_port', type=str, default=None,
                    help='串口设备路径，例如 /dev/ttyS0 或 /dev/ttyUSB0；为空则不启用串口')
    ap.add_argument('--serial_baud', type=int, default=460800,
                    help='串口波特率，协议要求 460800')

    # OSD / 交互 / 预测
    ap.add_argument('--no_osd', action='store_true', help='关闭 OSD 叠加，仅输出原始画面')
    ap.add_argument('--use_kalman', action='store_true', help='启用卡尔曼滤波平滑目标中心')
    ap.add_argument('--manual_select', action='store_true', help='允许在窗口中按 m 进入手动框选目标')

    # Web 手动框选（不影响 RTSP）
    ap.add_argument('--manual_select_web', action='store_true',
                    help='启用基于 Web 的手动框选（/manual 页面），不影响原有 RTSP 推流')

    return ap


def _init_serial(args):
    """根据命令行参数初始化串口，失败时仅打印警告并返回 None。"""
    if not args.serial_port:
        return None
    try:
        comm = SerialComm(port=args.serial_port, baudrate=args.serial_baud, auto_reconnect=True)
        if comm.connect():
            print(f"[serial] connected to {args.serial_port} @ {args.serial_baud}")
            return comm
        print(f"[serial] connect failed: {args.serial_port}")
    except Exception as e:
        print(f"[serial] init error: {e}")
    return None


def _send_tracking(comm, ts, frame_idx, center, bbox, status, confidence):
    """向串口发送当前帧的跟踪数据。

    - center: (cx, cy) 图像中心
    - bbox:   (x, y, w, h) 目标框；None 表示无目标
    - status: TrackingStatus
    """
    if comm is None:
        return

    cx, cy = center
    if bbox:
        x, y, w, h = bbox
        tx = x + w // 2
        ty = y + h // 2
        offx = tx - cx
        offy = ty - cy
    else:
        x = y = w = h = 0
        offx = offy = 0

    data = TrackingData(
        timestamp=float(ts),
        status=status,
        target_x=int(x),
        target_y=int(y),
        target_w=int(w),
        target_h=int(h),
        offset_x=int(offx),
        offset_y=int(offy),
        confidence=float(confidence or 0.0),
    )
    comm.send_tracking_data(data)


def main():
    args = build_arg_parser().parse_args()

    # 若启用 Web 手动框选，则启动 Flask 服务（守护线程），并注入共享状态
    if args.manual_select_web:
        set_shared_state(
            latest_ir_frame_ref,
            latest_tv_frame_ref,
            manual_bbox_ir_ref,
            manual_bbox_tv_ref,
        )
        run_app_in_thread(host="0.0.0.0", port=5000)
        print("[web] manual control enabled: http://<board-ip>:5000/manual")

    # 串口（可选）：仅在提供 serial_port 时启用
    serial_comm_ir = serial_comm_tv = None
    if args.serial_port:
        # 双路共用一个串口，将 IR/TV 信息按帧分别发送
        serial_comm_ir = _init_serial(args)
        serial_comm_tv = serial_comm_ir

    # 打开视频源（按模式）
    cap_ir = cap_tv = None
    if args.mode in ('ir', 'both'):
        cap_ir = open_camera(IR_DEVICE_ID, IR_WIDTH, IR_HEIGHT, IR_FPS,
                             use_pattern=args.pattern, name="IR")
        if not cap_ir:
            print("[fatal] IR source open failed.")
            sys.exit(1)
    if args.mode in ('tv', 'both'):
        cap_tv = open_camera(TV_DEVICE_ID, TV_WIDTH, TV_HEIGHT, TV_FPS,
                             use_pattern=args.pattern, name="TV")
        if not cap_tv:
            print("[fatal] TV source open failed.")
            sys.exit(1)

    # 加载模型
    mod_ir = model_ir = plat_ir = None
    mod_tv = model_tv = plat_tv = None
    if cap_ir:
        mod_ir, model_ir, plat_ir = setup_model_with_fallback(
            args.ir_model, branches=3, img_size=args.ir_img_size, quiet=args.quiet
        )
    if cap_tv:
        mod_tv, model_tv, plat_tv = setup_model_with_fallback(
            args.tv_model, branches=3, img_size=args.tv_img_size, quiet=args.quiet
        )

    helper_ir = COCO_test_helper(enable_letter_box=True) if cap_ir else None
    helper_tv = COCO_test_helper(enable_letter_box=True) if cap_tv else None

    # RTSP（可选）
    pusher_ir = pusher_tv = None
    if args.rtsp_ir or args.rtsp_tv:
        try:
            from tools.rtsp_push_appsrc import run_server
            if args.rtsp_ir and cap_ir:
                pusher_ir = run_server(IR_WIDTH, IR_HEIGHT, IR_FPS, args.rtsp_ir, args.rtsp_encoder, args.rtsp_bitrate)
            if args.rtsp_tv and cap_tv:
                pusher_tv = run_server(TV_WIDTH, TV_HEIGHT, TV_FPS, args.rtsp_tv, args.rtsp_encoder, args.rtsp_bitrate)
        except Exception as e:
            print("[rtsp] disabled:", e)

    # 跟踪与统计
    tracker_ir = tracker_tv = None
    tracking_ir = tracking_tv = False
    bbox_ir = bbox_tv = None
    score_ir = score_tv = 0.0

    center_ir = (IR_WIDTH // 2, IR_HEIGHT // 2)
    center_tv = (TV_WIDTH // 2, TV_HEIGHT // 2)

    # 合并 CSV
    os.makedirs(osp.dirname(args.save_csv), exist_ok=True)
    fout_all = open(args.save_csv, 'w')
    fout_all.write("ts,frame_idx,ir_ok,tv_ok,ir_bbox,ir_offx,ir_offy,tv_bbox,tv_offx,tv_offy\n")

    # 分路 CSV（可选）
    fout_ir = fout_tv = None
    if args.save_csv_ir:
        os.makedirs(osp.dirname(args.save_csv_ir), exist_ok=True)
        fout_ir = open(args.save_csv_ir, 'w')
        fout_ir.write("ts,frame_idx,ok,bbox,offx,offy\n")
    if args.save_csv_tv:
        os.makedirs(osp.dirname(args.save_csv_tv), exist_ok=True)
        fout_tv = open(args.save_csv_tv, 'w')
        fout_tv.write("ts,frame_idx,ok,bbox,offx,offy\n")

    frame_idx = 0
    t0 = time.time()
    print(f"[run] start: mode={args.mode}, IR={getattr(mod_ir,'IMG_SIZE',None)}, TV={getattr(mod_tv,'IMG_SIZE',None)}")

    # OSD 与交互
    osd_ir = osd_tv = None
    selector = None
    if not args.no_osd:
        osd_ir = OSDOverlay(language="zh")
        osd_tv = OSDOverlay(language="zh")
    if not args.headless and args.manual_select:
        selector = InteractiveSelector()

    # 卡尔曼滤波（针对目标中心）
    kf_ir = kf_tv = None
    if args.use_kalman:
        kf_ir = KalmanFilter(dt=1.0)
        kf_tv = KalmanFilter(dt=1.0)

    try:
        while True:
            ok_ir = ok_tv = False
            frm_ir = frm_tv = None
            if cap_ir:
                ok_ir, frm_ir = cap_ir.read()
            if cap_tv:
                ok_tv, frm_tv = cap_tv.read()

            if not ok_ir and not ok_tv:
                print("[run] both sources read failed or ended, exit loop.")
                break

            ts = time.time()

            # 更新 Web 预览帧（若启用 manual_select_web 时使用，不影响 RTSP）
            if ok_ir and frm_ir is not None:
                latest_ir_frame_ref["frame"] = frm_ir
            if ok_tv and frm_tv is not None:
                latest_tv_frame_ref["frame"] = frm_tv

            # IR 分支（独立跳帧）
            if cap_ir and ok_ir:
                if not tracking_ir:
                    do_det = True
                    if args.frame_skip_ir and args.frame_skip_ir > 1:
                        do_det = (frame_idx % args.frame_skip_ir == 0)
                    if do_det:
                        boxes, cls, sc = infer_once(mod_ir, model_ir, plat_ir, frm_ir, helper_ir)
                        bb = pick_primary_target(boxes, sc)
                        score_ir = float(np.max(sc)) if sc is not None and len(sc) > 0 else 0.0
                        if bb:
                            try:
                                tracker_ir = create_tracker()
                                tracker_ir.init(frm_ir, bb)
                                tracking_ir = True
                                bbox_ir = bb
                            except Exception as e:
                                print("[IR] tracker init failed:", e)
                                tracking_ir = False
                                tracker_ir = None
                else:
                    suc, bb = tracker_ir.update(frm_ir)
                    bbox_ir = tuple(map(int, bb)) if suc else None
                    tracking_ir = suc

                # 卡尔曼更新 / 预测（使用目标中心）
                if args.use_kalman and bbox_ir:
                    x, y, w, h = bbox_ir
                    cx_t, cy_t = x + w // 2, y + h // 2
                    if not kf_ir.initialized:
                        kf_ir.initialize(cx_t, cy_t)
                    else:
                        kf_ir.predict()
                        kf_ir.update(cx_t, cy_t)

                # OSD / 显示
                if not args.headless:
                    disp_ir = frm_ir.copy()
                    if not args.no_osd and osd_ir is not None:
                        status_str = "tracking" if bbox_ir else "lost"
                        # 若启用卡尔曼，则使用滤波后的中心计算偏移
                        off_ir_tmp = center_offset(center_ir, bbox_ir)
                        if args.use_kalman and kf_ir and kf_ir.initialized:
                            kx, ky = kf_ir.x[0, 0], kf_ir.x[1, 0]
                            offx_k, offy_k = int(kx - center_ir[0]), int(ky - center_ir[1])
                            off_ir_tmp = (offx_k, offy_k)
                        osd_ir.draw_complete_osd(
                            disp_ir,
                            bbox=bbox_ir,
                            offset=off_ir_tmp if bbox_ir else None,
                            status=status_str,
                            fps=float(frame_idx / max(1e-6, time.time() - t0)),
                            frame_idx=frame_idx,
                            mode="IR",
                        )
                    elif bbox_ir:
                        x, y, w, h = bbox_ir
                        cv2.rectangle(disp_ir, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("IR", disp_ir)
                if pusher_ir:
                    pusher_ir.push(disp_ir if not args.headless else frm_ir)

                # Web 手动框选：若收到来自 /manual_ir 的 bbox，则立即重置 IR tracker
                if args.manual_select_web and manual_bbox_ir_ref.get("bbox") is not None:
                    bbox_manual_ir = manual_bbox_ir_ref["bbox"]
                    manual_bbox_ir_ref["bbox"] = None
                    if bbox_manual_ir and frm_ir is not None:
                        bbox_ir = bbox_manual_ir
                        try:
                            tracker_ir = create_tracker()
                            tracker_ir.init(frm_ir, bbox_ir)
                            tracking_ir = True
                            print("[IR] manual bbox applied via web:", bbox_ir)
                        except Exception as e:
                            print("[IR] manual tracker init failed (web):", e)
                            tracking_ir = False
                            tracker_ir = None
                        if args.use_kalman and kf_ir is not None and bbox_ir is not None:
                            x, y, w, h = bbox_ir
                            kf_ir.initialize(x + w // 2, y + h // 2)

            # TV 分支（独立跳帧）
            if cap_tv and ok_tv:
                if not tracking_tv:
                    do_det = True
                    if args.frame_skip_tv and args.frame_skip_tv > 1:
                        do_det = (frame_idx % args.frame_skip_tv == 0)
                    if do_det:
                        boxes, cls, sc = infer_once(mod_tv, model_tv, plat_tv, frm_tv, helper_tv)
                        bb = pick_primary_target(boxes, sc)
                        score_tv = float(np.max(sc)) if sc is not None and len(sc) > 0 else 0.0
                        if bb:
                            try:
                                tracker_tv = create_tracker()
                                tracker_tv.init(frm_tv, bb)
                                tracking_tv = True
                                bbox_tv = bb
                            except Exception as e:
                                print("[TV] tracker init failed:", e)
                                tracking_tv = False
                                tracker_tv = None
                else:
                    suc, bb = tracker_tv.update(frm_tv)
                    bbox_tv = tuple(map(int, bb)) if suc else None
                    tracking_tv = suc

                # 卡尔曼更新 / 预测
                if args.use_kalman and bbox_tv:
                    x, y, w, h = bbox_tv
                    cx_t, cy_t = x + w // 2, y + h // 2
                    if not kf_tv.initialized:
                        kf_tv.initialize(cx_t, cy_t)
                    else:
                        kf_tv.predict()
                        kf_tv.update(cx_t, cy_t)

                # OSD / 显示
                if not args.headless:
                    disp_tv = frm_tv.copy()
                    if not args.no_osd and osd_tv is not None:
                        status_str = "tracking" if bbox_tv else "lost"
                        off_tv_tmp = center_offset(center_tv, bbox_tv)
                        if args.use_kalman and kf_tv and kf_tv.initialized:
                            kx, ky = kf_tv.x[0, 0], kf_tv.x[1, 0]
                            offx_k, offy_k = int(kx - center_tv[0]), int(ky - center_tv[1])
                            off_tv_tmp = (offx_k, offy_k)
                        osd_tv.draw_complete_osd(
                            disp_tv,
                            bbox=bbox_tv,
                            offset=off_tv_tmp if bbox_tv else None,
                            status=status_str,
                            fps=float(frame_idx / max(1e-6, time.time() - t0)),
                            frame_idx=frame_idx,
                            mode="TV",
                        )
                    elif bbox_tv:
                        x, y, w, h = bbox_tv
                        cv2.rectangle(disp_tv, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imshow("TV", disp_tv)
                if pusher_tv:
                    pusher_tv.push(disp_tv if not args.headless else frm_tv)

                # Web 手动框选：若收到来自 /manual_tv 的 bbox，则立即重置 tracker
                if args.manual_select_web and manual_bbox_tv_ref.get("bbox") is not None:
                    bbox_manual = manual_bbox_tv_ref["bbox"]
                    manual_bbox_tv_ref["bbox"] = None
                    if bbox_manual and frm_tv is not None:
                        bbox_tv = bbox_manual
                        try:
                            tracker_tv = create_tracker()
                            tracker_tv.init(frm_tv, bbox_tv)
                            tracking_tv = True
                            print("[TV] manual bbox applied via web:", bbox_tv)
                        except Exception as e:
                            print("[TV] manual tracker init failed (web):", e)
                            tracking_tv = False
                            tracker_tv = None
                        if args.use_kalman and kf_tv is not None and bbox_tv is not None:
                            x, y, w, h = bbox_tv
                            kf_tv.initialize(x + w // 2, y + h // 2)

            # 偏移统计（串口与 CSV 使用同一套数据；若启用卡尔曼则用滤波中心）
            off_ir = center_offset(center_ir, bbox_ir) if cap_ir else (None, None)
            off_tv = center_offset(center_tv, bbox_tv) if cap_tv else (None, None)
            if args.use_kalman and kf_ir and kf_ir.initialized and cap_ir and bbox_ir:
                kx, ky = kf_ir.x[0, 0], kf_ir.x[1, 0]
                off_ir = (int(kx - center_ir[0]), int(ky - center_ir[1]))
            if args.use_kalman and kf_tv and kf_tv.initialized and cap_tv and bbox_tv:
                kx, ky = kf_tv.x[0, 0], kf_tv.x[1, 0]
                off_tv = (int(kx - center_tv[0]), int(ky - center_tv[1]))

            # 串口发送（若启用）
            if serial_comm_ir and cap_ir:
                st_ir = TrackingStatus.TRACKING if bbox_ir else TrackingStatus.LOST
                _send_tracking(serial_comm_ir, ts, frame_idx, center_ir, bbox_ir, st_ir, score_ir)
            if serial_comm_tv and cap_tv:
                st_tv = TrackingStatus.TRACKING if bbox_tv else TrackingStatus.LOST
                _send_tracking(serial_comm_tv, ts, frame_idx, center_tv, bbox_tv, st_tv, score_tv)

            # 合并 CSV
            fout_all.write(
                f"{ts:.6f},{frame_idx},"
                f"{int(bool(ok_ir))},{int(bool(ok_tv))},"
                f"{bbox_ir},{'' if off_ir[0] is None else off_ir[0]},"
                f"{'' if off_ir[1] is None else off_ir[1]},"
                f"{bbox_tv},{'' if off_tv[0] is None else off_tv[0]},"
                f"{'' if off_tv[1] is None else off_tv[1]}\n"
            )

            # 分路 CSV（可选）
            if fout_ir and cap_ir:
                fout_ir.write(
                    f"{ts:.6f},{frame_idx},{int(bool(ok_ir))},{bbox_ir},"
                    f"{'' if off_ir[0] is None else off_ir[0]},"
                    f"{'' if off_ir[1] is None else off_ir[1]}\n"
                )
            if fout_tv and cap_tv:
                fout_tv.write(
                    f"{ts:.6f},{frame_idx},{int(bool(ok_tv))},{bbox_tv},"
                    f"{'' if off_tv[0] is None else off_tv[0]},"
                    f"{'' if off_tv[1] is None else off_tv[1]}\n"
                )

            frame_idx += 1
            if args.quiet:
                el = max(1e-6, time.time() - t0)
                fps = frame_idx / el
                print(f"\r[run] frames={frame_idx} FPS≈{fps:5.1f}", end="", flush=True)

            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[run] user pressed 'q', exit.")
                    break
                # 手动框选目标（仅在启用 manual_select 时响应）
                if selector is not None and key == ord('m'):
                    # 优先使用 TV 画面选目标，其次 IR
                    src_frame = None
                    if cap_tv and ok_tv:
                        src_frame = frm_tv
                    elif cap_ir and ok_ir:
                        src_frame = frm_ir
                    if src_frame is not None:
                        bbox_manual = selector.select_by_drag(src_frame)
                        if bbox_manual:
                            # 重置跟踪器和卡尔曼
                            if cap_tv and ok_tv:
                                bbox_tv = bbox_manual
                                try:
                                    tracker_tv = create_tracker()
                                    tracker_tv.init(src_frame, bbox_tv)
                                    tracking_tv = True
                                except Exception as e:
                                    print("[TV] manual tracker init failed:", e)
                            elif cap_ir and ok_ir:
                                bbox_ir = bbox_manual
                                try:
                                    tracker_ir = create_tracker()
                                    tracker_ir.init(src_frame, bbox_ir)
                                    tracking_ir = True
                                except Exception as e:
                                    print("[IR] manual tracker init failed:", e)
                            if args.use_kalman:
                                if cap_tv and ok_tv and bbox_tv:
                                    x, y, w, h = bbox_tv
                                    kf_tv.initialize(x + w // 2, y + h // 2)
                                if cap_ir and ok_ir and bbox_ir:
                                    x, y, w, h = bbox_ir
                                    kf_ir.initialize(x + w // 2, y + h // 2)
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"\n[auto-exit] reach max_frames={args.max_frames}")
                break
    finally:
        if cap_ir: cap_ir.release()
        if cap_tv: cap_tv.release()
        if not args.headless:
            cv2.destroyAllWindows()
        try:
            if model_ir: model_ir.release()
        except Exception:
            pass
        try:
            if model_tv: model_tv.release()
        except Exception:
            pass
        fout_all.close()
        if fout_ir: fout_ir.close()
        if fout_tv: fout_tv.close()
        if serial_comm_ir:
            try:
                serial_comm_ir.disconnect()
            except Exception:
                pass
        if args.quiet:
            print("\n[run] done.")


if __name__ == "__main__":
    main()