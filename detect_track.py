import os
import sys
import argparse
import os.path as osp
from types import SimpleNamespace

import cv2
import numpy as np
import time

from py_utils.coco_utils import COCO_test_helper

# 为不同模型准备独立的 yolov8 模块（方案4）
# 优先使用 yolov8_ir / yolov8_tv；若不存在则回退到 yolov8（向后兼容）
y8_ir = None
y8_tv = None
y8_default = None
try:
    import yolov8_ir as y8_ir
except Exception:
    y8_ir = None
try:
    import yolov8_tv as y8_tv
except Exception:
    y8_tv = None
try:
    import yolov8 as y8_default  # 旧版/通用模块
except Exception:
    y8_default = None


def _select_y8_module(model_path: str):
    """
    根据模型文件名选择对应的 yolov8 模块：
    - 文件名包含 'ir' -> yolov8_ir
    - 文件名包含 'tv' -> yolov8_tv
    - 否则回退 yolov8（若可用），再退回 yolov8_ir（最后兜底）
    """
    name = (osp.basename(model_path) if model_path else "").lower()
    if 'ir' in name and y8_ir is not None:
        return y8_ir, 'ir'
    if 'tv' in name and y8_tv is not None:
        return y8_tv, 'tv'
    if y8_default is not None:
        return y8_default, 'default'
    # 兜底：至少返回一个模块，避免崩溃
    if y8_ir is not None:
        return y8_ir, 'ir'
    if y8_tv is not None:
        return y8_tv, 'tv'
    raise RuntimeError("No yolov8 module available: please provide yolov8_ir.py / yolov8_tv.py or yolov8.py.")


# =============== 用户可编辑区域（无需记命令，改这里就行） ===============
CLASSES_TXT = 'classes.txt'  # 或 None
BRANCHES = 3
TARGET = 'rk3588'
DEVICE_ID = None
IMG_SHOW = False
SAVE_VIDEO = True
SAVE_IMAGE = True
RESULT_ROOT = 'my_yolo/results/detect_track'
QUIET_DEFAULT = True

# 多任务示例（支持 IR/TV 各自尺寸，自动匹配 yolov8_ir/yolov8_tv）
TASKS = [
    {
        'model_path': 'my_yolo/weights/IR.rknn',
        'img_folder': 'my_yolo/datasets/images/IR',
        'img_size': '640x640',
        'enabled': True,
        'img_save': True,
        'result_tag': 'IR_images',
    },
    {
        'model_path': 'my_yolo/weights/TV.rknn',
        'img_folder': 'my_yolo/datasets/images/TV',
        'img_size': '1088x1088',
        'enabled': True,
        'img_save': True,
        'result_tag': 'TV_images',
    },
    {
        'model_path': 'my_yolo/weights/IR.rknn',
        'video_file': 'my_yolo/datasets/video/IR/IR.mp4',
        'img_size': '640x640',
        'enabled': True,
        'save_video': True,
        'frame_skip': 0,
        'show_offset': True,
        'result_tag': 'IR_video',
        'keep_name': True,
    },
    {
        'model_path': 'my_yolo/weights/TV.rknn',
        'video_folder': 'my_yolo/datasets/video/TV/TV.mp4',
        'img_size': '1088x1088',
        'enabled': True,
        'save_video': True,
        'frame_skip': 0,
        'show_offset': True,
        'result_tag': 'TV_video',
        'keep_name': True,
    },
    {
        'model_path': 'my_yolo/weights/xx.rknn',
        'video_folder': 'my_yolo/datasets/video',
        'img_size': '640x640',
        'enabled': False,
        'save_video': True,
        'frame_skip': 0,
        'show_offset': False,
        'result_tag': 'multi_video',
    },
]
# ===============================================================


def build_arg_parser():
    p = argparse.ArgumentParser(description='Batch video tracking (RKNN/ONNX/PT detection + OpenCV CSRT, no torch)')
    p.add_argument('--classes_txt', type=str, default=CLASSES_TXT)
    p.add_argument('--branches', type=int, default=BRANCHES)
    p.add_argument('--target', type=str, default=TARGET)
    p.add_argument('--device_id', type=str, default=DEVICE_ID)

    group_show = p.add_mutually_exclusive_group(required=False)
    group_show.add_argument('--img_show', dest='img_show', action='store_true')
    group_show.add_argument('--no-img-show', dest='img_show', action='store_false')
    p.set_defaults(img_show=IMG_SHOW)

    group_save = p.add_mutually_exclusive_group(required=False)
    group_save.add_argument('--save_video', dest='save_video', action='store_true')
    group_save.add_argument('--no-save-video', dest='save_video', action='store_false')
    p.set_defaults(save_video=SAVE_VIDEO)

    group_imgsv = p.add_mutually_exclusive_group(required=False)
    group_imgsv.add_argument('--img_save', dest='img_save', action='store_true')
    group_imgsv.add_argument('--no-img-save', dest='img_save', action='store_false')
    p.set_defaults(img_save=SAVE_IMAGE)

    group_quiet = p.add_mutually_exclusive_group(required=False)
    group_quiet.add_argument('--quiet', dest='quiet', action='store_true')
    group_quiet.add_argument('--no-quiet', dest='quiet', action='store_false')
    p.set_defaults(quiet=QUIET_DEFAULT)

    # 单次运行（可选）
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--video_path', type=str, default=None)
    p.add_argument('--video_dir', type=str, default=None)
    p.add_argument('--img_folder', type=str, default=None)
    p.add_argument('--img_size', type=str, default=None)
    p.add_argument('--video_fps', type=float, default=None)
    p.add_argument('--frame_skip', type=int, default=0)
    p.add_argument('--result_tag', type=str, default=None)
    p.add_argument('--result_root', type=str, default=RESULT_ROOT)
    p.add_argument('--show_offset', action='store_true')
    p.add_argument('--keep_name', action='store_true')
    return p


def _parse_img_size(s):
    s = s.lower().replace('×', 'x').replace(',', 'x')
    parts = s.split('x')
    if len(parts) == 1:
        v = int(parts[0]); return (v, v)
    return (int(parts[0]), int(parts[1]))


def setup_model_and_env(y8m, args):
    """
    将原先使用全局 y8 的逻辑参数化为 y8m（对应 yolov8_ir / yolov8_tv / yolov8）。
    """
    y8m.OBJ_THRESH = float(getattr(args, 'obj_thresh', 0.25))
    y8m.NMS_THRESH = float(getattr(args, 'nms_thresh', 0.45))
    y8m.BRANCHES = int(args.branches)

    if args.classes_txt and osp.exists(args.classes_txt):
        with open(args.classes_txt, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]
        if names:
            y8m.CLASSES = tuple(names)
            y8m.coco_id_list = list(range(1, len(names) + 1))

    if args.img_size:
        w, h = _parse_img_size(args.img_size)
        y8m.IMG_SIZE = (int(w), int(h))

    y8m.QUIET = bool(getattr(args, 'quiet', False))

    ns = SimpleNamespace(
        model_path=args.model_path,
        target=args.target,
        device_id=args.device_id,
        quiet=y8m.QUIET,
    )
    model, platform = y8m.setup_model(ns)

    if not args.img_size and platform == 'rknn':
        try:
            ops = model.rknn.get_input_ops()
            if ops and 'shape' in ops[0]:
                shp = ops[0]['shape']
                if len(shp) == 4:
                    if shp[1] in (1, 3):
                        y8m.INPUT_LAYOUT = 'nchw'
                        y8m.IMG_SIZE = (int(shp[3]), int(shp[2]))
                    elif shp[-1] in (1, 3):
                        y8m.INPUT_LAYOUT = 'nhwc'
                        y8m.IMG_SIZE = (int(shp[2]), int(shp[1]))
        except Exception:
            pass

    return model, platform


def infer_once(y8m, model, platform, frame_bgr, helper):
    img = helper.letter_box(im=frame_bgr.copy(), new_shape=(y8m.IMG_SIZE[1], y8m.IMG_SIZE[0]), pad_color=(0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if platform in ['pytorch', 'onnx']:
        input_data = img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.0
    else:
        if y8m.INPUT_LAYOUT == 'nchw':
            input_data = img.transpose(2, 0, 1)[None, ...].astype(np.uint8)
        else:
            input_data = np.expand_dims(img, 0).astype(np.uint8)

    try:
        outputs = model.run([input_data])
    except Exception as e:
        print(f"[infer] error: {e}, shape={getattr(input_data, 'shape', None)}, dtype={getattr(input_data, 'dtype', None)}")
        return None, None, None

    if outputs is None or (isinstance(outputs, list) and len(outputs) == 0):
        return None, None, None

    boxes, classes, scores = y8m.post_process(outputs)
    if boxes is None:
        return None, None, None
    real_boxes = helper.get_real_box(boxes)
    return real_boxes, classes, scores


def _is_image_file(path):
    lower = path.lower()
    return any(lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp'))


def pick_primary_target(boxes, scores):
    if boxes is None or scores is None or len(scores) == 0:
        return None
    idx = int(np.argmax(scores))
    x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return (x1, y1, w, h)


def ensure_video_writer(out_dir, base_name, fps, width, height):
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"{base_name}_track.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    return writer, out_path


def draw_offset_info(frame, center_xy, bbox_xywh):
    if bbox_xywh is None:
        return
    cx, cy = center_xy
    x, y, w, h = bbox_xywh
    tx, ty = x + w // 2, y + h // 2
    off_x, off_y = tx - cx, ty - cy
    wdt = frame.shape[1]
    cv2.putText(frame, f"Offset: X={int(off_x):4d}, Y={int(off_y):4d}", (max(10, wdt - 300), 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    cv2.circle(frame, (tx, ty), 5, (0, 0, 255), -1)
    cv2.line(frame, (cx, cy), (tx, ty), (255, 255, 0), 1)


def _create_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    raise RuntimeError('Your OpenCV build lacks CSRT/KCF trackers. Please install opencv-contrib.')


def process_single_video(y8m, args, model, platform, vpath):
    if not osp.exists(vpath):
        print(f"[video] not found: {vpath}")
        return
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"[video] open failed: {vpath}")
        return

    helper = COCO_test_helper(enable_letter_box=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = args.video_fps if args.video_fps and args.video_fps > 0 else in_fps
    center_xy = (width // 2, height // 2)

    tag = args.result_tag if args.result_tag else osp.splitext(osp.basename(args.model_path))[0]
    out_root = osp.join(args.result_root, 'video', tag)
    base = osp.splitext(osp.basename(vpath))[0]
    writer = None
    out_path = None
    if args.save_video:
        if getattr(args, 'keep_name', False):
            os.makedirs(out_root, exist_ok=True)
            out_path = osp.join(out_root, f"{base}.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))
        else:
            writer, out_path = ensure_video_writer(out_root, base, out_fps, width, height)

    tracker = None
    tracking = False
    bbox_xywh = None
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    t0 = time.time()

    def _print_progress():
        if not getattr(args, 'quiet', False):
            return
        elapsed = max(1e-6, time.time() - t0)
        fps = frame_idx / elapsed if frame_idx > 0 else 0.0
        if total_frames > 0:
            pct = 100.0 * frame_idx / max(1, total_frames)
            msg = f"\r[track] {pct:5.1f}% | {frame_idx}/{total_frames} | FPS: {fps:5.1f}"
        else:
            msg = f"\r[track] frames {frame_idx} | FPS: {fps:5.1f}"
        print(msg, end='', flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if not tracking:
                do_detect = True
                if args.frame_skip and args.frame_skip > 1:
                    do_detect = ((frame_idx - 1) % int(args.frame_skip) == 0)
                if do_detect:
                    boxes, classes, scores = infer_once(y8m, model, platform, frame, helper)
                    bbox = pick_primary_target(boxes, scores)
                    if bbox is not None:
                        tracker = _create_tracker()
                        tracker.init(frame, tuple(bbox))
                        tracking = True
                        bbox_xywh = bbox
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, 'Detection', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    bbox_xywh = (x, y, w, h)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Tracking', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                else:
                    tracking = False
                    tracker = None
                    cv2.putText(frame, 'Tracking Lost', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if args.show_offset and bbox_xywh is not None:
                draw_offset_info(frame, center_xy, bbox_xywh)

            if writer is not None:
                writer.write(frame)

            if args.img_show:
                cv2.imshow('detect_track', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            _print_progress()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.img_show:
            cv2.destroyAllWindows()

    if getattr(args, 'quiet', False):
        print()
    if args.save_video and out_path and not getattr(args, 'quiet', False):
        print(f"[save] {out_path}")
    elif args.save_video and out_path and getattr(args, 'quiet', False):
        print(f"[done] saved: {out_path}")


def process_image_folder(y8m, args, model, platform, img_dir):
    if not osp.isdir(img_dir):
        print(f"[images] not a directory: {img_dir}")
        return
    files = sorted(os.listdir(img_dir))
    images = [f for f in files if _is_image_file(f)]
    if not images:
        print(f"[images] no images under: {img_dir}")
        return
    helper = COCO_test_helper(enable_letter_box=True)

    tag = args.result_tag if args.result_tag else osp.splitext(osp.basename(args.model_path))[0]
    out_root = osp.join(args.result_root, 'images', tag)
    if args.img_save:
        os.makedirs(out_root, exist_ok=True)

    total = len(images)

    def _print_progress(i):
        if not getattr(args, 'quiet', False):
            return
        pct = 100.0 * (i + 1) / max(1, total)
        print(f"\r[images] {pct:5.1f}% | {i+1}/{total}", end='', flush=True)

    for i, name in enumerate(images):
        ipath = osp.join(img_dir, name)
        img = cv2.imread(ipath)
        if img is None:
            continue
        boxes, classes, scores = infer_once(y8m, model, platform, img, helper)
        draw_img = img.copy()
        if boxes is not None:
            y8m.draw(draw_img, boxes, scores, classes)
        if args.img_save:
            opath = osp.join(out_root, name)
            cv2.imwrite(opath, draw_img)
        if args.img_show:
            cv2.imshow('detect_track_images', draw_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        _print_progress(i)
    if args.img_show:
        cv2.destroyAllWindows()
    if getattr(args, 'quiet', False):
        print()


def prepare_env_for_task(y8m, args, model, platform):
    y8m.OBJ_THRESH = float(getattr(args, 'obj_thresh', 0.25))
    y8m.NMS_THRESH = float(getattr(args, 'nms_thresh', 0.45))
    y8m.BRANCHES = int(args.branches)

    if args.classes_txt and osp.exists(args.classes_txt):
        with open(args.classes_txt, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]
        if names:
            y8m.CLASSES = tuple(names)
            y8m.coco_id_list = list(range(1, len(names) + 1))

    if args.img_size:
        w, h = _parse_img_size(args.img_size)
        y8m.IMG_SIZE = (int(w), int(h))

    y8m.QUIET = bool(getattr(args, 'quiet', False))

    if not args.img_size and platform == 'rknn':
        try:
            ops = model.rknn.get_input_ops()
            if ops and 'shape' in ops[0]:
                shp = ops[0]['shape']
                if len(shp) == 4:
                    if shp[1] in (1, 3):
                        y8m.INPUT_LAYOUT = 'nchw'
                        y8m.IMG_SIZE = (int(shp[3]), int(shp[2]))
                    elif shp[-1] in (1, 3):
                        y8m.INPUT_LAYOUT = 'nhwc'
                        y8m.IMG_SIZE = (int(shp[2]), int(shp[1]))
        except Exception:
            pass


def main():
    parser = build_arg_parser()
    cli = parser.parse_args()

    if cli.classes_txt is not None:
        s = str(cli.classes_txt).strip().lower()
        if s in ('none', 'null', ''):
            cli.classes_txt = None

    # 单次运行：构造一条任务，不走 TASKS
    ALL_TASKS = None
    if cli.model_path and (cli.video_path or cli.video_dir or cli.img_folder):
        single_task = {
            'model_path': cli.model_path,
            'enabled': True,
        }
        if cli.img_folder:
            single_task['img_folder'] = cli.img_folder
        else:
            if cli.video_path:
                single_task['video_file'] = cli.video_path
            else:
                single_task['video_folder'] = cli.video_dir
        if cli.img_size:
            single_task['img_size'] = cli.img_size
        if cli.result_tag:
            single_task['result_tag'] = cli.result_tag
        single_task['frame_skip'] = cli.frame_skip
        single_task['save_video'] = cli.save_video
        single_task['img_save'] = cli.img_save
        single_task['show_offset'] = cli.show_offset
        single_task['keep_name'] = cli.keep_name
        single_task['result_root'] = cli.result_root
        ALL_TASKS = [single_task]
    else:
        ALL_TASKS = TASKS

    failures = []

    # 按模型分组（与原逻辑一致）
    grouped = {}
    for t in ALL_TASKS:
        if 'results_tag' in t and 'result_tag' not in t:
            t['result_tag'] = t['results_tag']
        if not t.get('enabled', True):
            tag = t.get('result_tag')
            if not tag:
                if t.get('video_file'):
                    tag = osp.splitext(osp.basename(t['video_file']))[0]
                elif t.get('video_folder'):
                    p = t['video_folder']
                    tag = osp.splitext(osp.basename(p))[0] if osp.splitext(p)[1] else osp.basename(p)
                elif t.get('img_folder'):
                    tag = osp.basename(t['img_folder'].rstrip('/\\'))
                else:
                    tag = 'unknown'
            print(f"[skip] {tag} is disabled")
            continue
            # 注意：不同模型会被分到不同组，各组分别加载各自模块与模型
        model_path = t['model_path']
        grouped.setdefault(model_path, []).append(t)

    for model_path, tasks in grouped.items():
        if not osp.exists(model_path):
            print('[error] model not found:', model_path)
            for t in tasks:
                name = t.get('result_tag') or 'unknown'
                failures.append((name, 1))
            continue

        # 根据模型名选择对应 yolov8 模块
        y8m, mod_name = _select_y8_module(model_path)
        if not cli.quiet:
            print(f"\n[module] use {mod_name} for model: {model_path}")

        # 加载一次模型
        base_ns = SimpleNamespace(
            model_path=model_path,
            target=cli.target,
            device_id=cli.device_id,
            classes_txt=cli.classes_txt,
            branches=cli.branches,
            img_size=None,
            quiet=cli.quiet,
        )
        model, platform = setup_model_and_env(y8m, base_ns)

        try:
            for t in tasks:
                img_size = t.get('img_size')

                if t.get('img_folder'):
                    img_dir = t['img_folder']
                    if not osp.exists(img_dir):
                        print('[error] image folder not found:', img_dir)
                        failures.append((t.get('result_tag') or osp.basename(img_dir.rstrip('/\\')), 1))
                        continue
                    a = SimpleNamespace(
                        model_path=model_path,
                        target=cli.target,
                        device_id=cli.device_id,
                        classes_txt=cli.classes_txt,
                        branches=cli.branches,
                        img_size=img_size,
                        img_save=(t.get('img_save') if 'img_save' in t else cli.img_save),
                        img_show=cli.img_show,
                        result_tag=(t.get('result_tag') if t.get('result_tag') else cli.result_tag),
                        result_root=(t.get('result_root') if t.get('result_root') else cli.result_root),
                        quiet=(t.get('quiet') if 'quiet' in t else cli.quiet),
                        obj_thresh=0.25,
                        nms_thresh=0.45,
                    )
                    prepare_env_for_task(y8m, a, model, platform)
                    if not cli.quiet:
                        print(f"\n[run-images] model={model_path} images={img_dir}")
                    process_image_folder(y8m, a, model, platform, img_dir)
                else:
                    video_src = t.get('video_file') or t.get('video_folder')
                    if not osp.exists(video_src):
                        print('[error] video source not found:', video_src)
                        name = t.get('result_tag') or (
                            osp.splitext(osp.basename(video_src))[0] if t.get('video_file') else osp.basename(video_src)
                        )
                        failures.append((name, 1))
                        continue
                    a = SimpleNamespace(
                        model_path=model_path,
                        target=cli.target,
                        device_id=cli.device_id,
                        classes_txt=cli.classes_txt,
                        branches=cli.branches,
                        img_size=img_size,
                        video_fps=cli.video_fps,
                        frame_skip=(t.get('frame_skip') if 'frame_skip' in t else cli.frame_skip),
                        result_tag=(t.get('result_tag') if t.get('result_tag') else cli.result_tag),
                        result_root=(t.get('result_root') if 'result_root' in t else cli.result_root),
                        img_show=cli.img_show,
                        save_video=(t.get('save_video') if 'save_video' in t else cli.save_video),
                        show_offset=(t.get('show_offset') if 'show_offset' in t else cli.show_offset),
                        keep_name=(t.get('keep_name') if 'keep_name' in t else cli.keep_name),
                        quiet=(t.get('quiet') if 'quiet' in t else cli.quiet),
                        obj_thresh=0.25,
                        nms_thresh=0.45,
                    )
                    prepare_env_for_task(y8m, a, model, platform)
                    if not cli.quiet:
                        print(f"\n[run-track] model={model_path} video={video_src}")
                    process_single_video(y8m, a, model, platform, video_src)
        except Exception as e:
            print('[error] grouped tasks failed for model:', model_path, e)
            failures.append((osp.basename(model_path), 1))
        finally:
            model.release()

    if failures:
        print('\n[summary] some tracking tasks failed:')
        for n, c in failures:
            print(f'  - {n}: exitcode={c}')
        sys.exit(1)
    else:
        print('\n[summary] all tracking tasks finished successfully.')
        sys.exit(0)


if __name__ == '__main__':
    main()