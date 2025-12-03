import cv2
import sys
import argparse
import time

# ================= 配置区域 =================
# 根据实际情况修改两个设备号（现场接入相机后）
IR_DEVICE_ID = -1         # 红外相机 /dev/videoX（示例值）
TV_DEVICE_ID = -1         # 可见光相机 /dev/videoY（示例值）

# 红外相机参数（现场接入后按实际修改）
IR_WIDTH = 640
IR_HEIGHT = 512
IR_FPS = 50

# 可见光相机参数（现场接入后按实际修改）
TV_WIDTH = 1920
TV_HEIGHT = 1080
TV_FPS = 25

# GStreamer 尝试的像素格式优先级（从前到后依次尝试）
# 现场根据 v4l2-ctl --list-formats-ext 的真实支持情况调整顺序/内容
PREFERRED_FORMATS = ["NV12", "YUY2", "YUYV", "UYVY", "NV16"]
# ===========================================


def build_pipeline(device_id, width, height, fps, pix_fmt, use_pattern=False):
    """构建 GStreamer 管道字符串。
    - use_pattern=True 时使用 videotestsrc（无相机也能跑）
    - use_pattern=False 时使用 v4l2src（真实相机）
    """
    if use_pattern:
        # 测试图案源：给出明确的 caps，便于和 appsink 对接
        return (
            f"videotestsrc is-live=true pattern=ball ! "
            f"video/x-raw,format={pix_fmt},width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
    # 真实相机
    return (
        f"v4l2src device=/dev/video{device_id} ! "
        f"video/x-raw,format={pix_fmt},width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
    )


def try_open_camera(device_id, width, height, fps, use_pattern=False):
    """依次尝试多种像素格式，返回成功的 cap 及使用的像素格式。"""
    last_err = None
    for fmt in PREFERRED_FORMATS:
        pipeline = build_pipeline(device_id, width, height, fps, fmt, use_pattern=use_pattern)
        src_name = "videotestsrc" if use_pattern else f"/dev/video{device_id}"
        print(f"尝试使用像素格式 {fmt} 打开 {src_name} ...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            # 打印实际生效的分辨率和 FPS
            real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            real_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            print(
                f"成功：{src_name} 使用 {fmt} 打开，"
                f"实际分辨率 {int(real_w)}x{int(real_h)}，FPS≈{real_fps:.2f}"
            )
            return cap, fmt
        last_err = fmt
        cap.release()

    src_name = "videotestsrc" if use_pattern else f"/dev/video{device_id}"
    print(f"错误：无法使用 {PREFERRED_FORMATS} 中的任一格式打开 {src_name}")
    if last_err is not None and not use_pattern:
        print("请用 v4l2-ctl --list-formats-ext -d /dev/videoX 检查实际像素格式")
    return None, None


def show_single_camera(name, device_id, width, height, fps, use_pattern, headless, max_frames, save_first):
    cap, used_fmt = try_open_camera(device_id, width, height, fps, use_pattern=use_pattern)
    if cap is None:
        sys.exit(1)

    src_name = "videotestsrc" if use_pattern else f"/dev/video{device_id}"
    print(f"{name}（{src_name}，格式 {used_fmt}）打开成功，按 'q' 退出（有窗口时）...")

    frame_idx = 0
    t0 = time.time()
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"错误：{name} 无法读取帧数据")
            break

        frame_idx += 1

        if save_first and not saved:
            cv2.imwrite(save_first, frame)
            print(f"[saved] 第一帧已保存到: {save_first}")
            saved = True

        if not headless:
            cv2.imshow(name, frame)
            # 只有在有窗口的前提下，按键才有效；焦点需在窗口上
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if max_frames and frame_idx >= max_frames:
            print(f"[auto-exit] 已读取 {frame_idx} 帧，自动退出。")
            break

        # 简单的进度/FPS 打印
        if frame_idx % 30 == 0:
            elapsed = max(1e-6, time.time() - t0)
            fps_now = frame_idx / elapsed
            print(f"\r{ name }: frames={frame_idx} FPS≈{fps_now:4.1f}", end="", flush=True)

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print()  # 换行


def show_dual_camera(args):
    """同时打开 IR 与 TV 两路相机并分别显示。"""
    cap_ir, fmt_ir = try_open_camera(IR_DEVICE_ID, IR_WIDTH, IR_HEIGHT, IR_FPS, use_pattern=args.pattern)
    if cap_ir is None:
        print("红外相机打开失败，终止。")
        sys.exit(1)

    cap_tv, fmt_tv = try_open_camera(TV_DEVICE_ID, TV_WIDTH, TV_HEIGHT, TV_FPS, use_pattern=args.pattern)
    if cap_tv is None:
        print("可见光相机打开失败，终止。")
        cap_ir.release()
        sys.exit(1)

    print(
        f"IR({ 'videotestsrc' if args.pattern else f'/dev/video{IR_DEVICE_ID}' },{fmt_ir}) "
        f"与 TV({ 'videotestsrc' if args.pattern else f'/dev/video{TV_DEVICE_ID}' },{fmt_tv}) 已打开"
    )

    frame_idx = 0
    t0 = time.time()
    saved_ir = saved_tv = False

    while True:
        ret_ir, frame_ir = cap_ir.read()
        ret_tv, frame_tv = cap_tv.read()

        if not ret_ir and not ret_tv:
            print("两路相机均读取失败，退出。")
            break

        if args.save_first and not saved_ir and ret_ir:
            cv2.imwrite(args.save_first.replace(".","_ir."), frame_ir)
            print(f"[saved] IR 第一帧 -> {args.save_first.replace('.','_ir.')}")
            saved_ir = True
        if args.save_first and not saved_tv and ret_tv:
            cv2.imwrite(args.save_first.replace(".","_tv."), frame_tv)
            print(f"[saved] TV 第一帧 -> {args.save_first.replace('.','_tv.')}")
            saved_tv = True

        if not args.headless:
            if ret_ir:
                cv2.imshow("IR Camera", frame_ir)
            if ret_tv:
                cv2.imshow("TV Camera", frame_tv)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            print(f"[auto-exit] 已读取 {frame_idx} 帧，自动退出。")
            break

        if frame_idx % 30 == 0:
            elapsed = max(1e-6, time.time() - t0)
            fps_now = frame_idx / elapsed
            print(f"\rdual: frames={frame_idx} FPS≈{fps_now:4.1f}", end="", flush=True)

    cap_ir.release()
    cap_tv.release()
    if not args.headless:
        cv2.destroyAllWindows()
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="RK3588 MIPI 双路相机测试脚本")
    parser.add_argument(
        "--mode",
        choices=["ir", "tv", "both"],
        default="ir",
        help="选择测试模式：ir=仅红外，tv=仅可见光，both=同时两路",
    )
    # 新增：无相机也能测试
    parser.add_argument("--pattern", action="store_true", help="使用测试图案源(videotestsrc)而非真实相机")
    parser.add_argument("--headless", action="store_true", help="无界面模式（不弹窗口）")
    parser.add_argument("--max-frames", type=int, default=0, help="最多读取多少帧后自动退出（0 表示不限制）")
    parser.add_argument("--save-first", type=str, default=None, help="把第一帧保存到此路径（例如 /tmp/first.jpg）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "ir":
        show_single_camera(
            "IR Camera", IR_DEVICE_ID, IR_WIDTH, IR_HEIGHT, IR_FPS,
            use_pattern=args.pattern, headless=args.headless,
            max_frames=args.max_frames, save_first=args.save_first
        )
    elif args.mode == "tv":
        show_single_camera(
            "TV Camera", TV_DEVICE_ID, TV_WIDTH, TV_HEIGHT, TV_FPS,
            use_pattern=args.pattern, headless=args.headless,
            max_frames=args.max_frames, save_first=args.save_first
        )
    else:
        show_dual_camera(args)


if __name__ == "__main__":
    main()