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

PREFERRED_FORMATS = ["YUY2", "YUYV", "UYVY", "NV16", "NV12"]


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
    if y8_default is not None:
        return y8_default, 'default'
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

    return ap


def main():
    args = build_arg_parser().parse_args()

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

            # IR 分支（独立跳帧）
            if cap_ir and ok_ir:
                if not tracking_ir:
                    do_det = True
                    if args.frame_skip_ir and args.frame_skip_ir > 1:
                        do_det = (frame_idx % args.frame_skip_ir == 0)
                    if do_det:
                        boxes, cls, sc = infer_once(mod_ir, model_ir, plat_ir, frm_ir, helper_ir)
                        bb = pick_primary_target(boxes, sc)
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

                if not args.headless and bbox_ir:
                    x, y, w, h = bbox_ir
                    cv2.rectangle(frm_ir, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("IR", frm_ir)
                if pusher_ir:
                    pusher_ir.push(frm_ir)

            # TV 分支（独立跳帧）
            if cap_tv and ok_tv:
                if not tracking_tv:
                    do_det = True
                    if args.frame_skip_tv and args.frame_skip_tv > 1:
                        do_det = (frame_idx % args.frame_skip_tv == 0)
                    if do_det:
                        boxes, cls, sc = infer_once(mod_tv, model_tv, plat_tv, frm_tv, helper_tv)
                        bb = pick_primary_target(boxes, sc)
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

                if not args.headless and bbox_tv:
                    x, y, w, h = bbox_tv
                    cv2.rectangle(frm_tv, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imshow("TV", frm_tv)
                if pusher_tv:
                    pusher_tv.push(frm_tv)

            # 偏移统计
            off_ir = center_offset(center_ir, bbox_ir) if cap_ir else (None, None)
            off_tv = center_offset(center_tv, bbox_tv) if cap_tv else (None, None)

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
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[run] user pressed 'q', exit.")
                    break
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
        if args.quiet:
            print("\n[run] done.")


if __name__ == "__main__":
    main()