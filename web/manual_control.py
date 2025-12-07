import threading
from typing import Optional, Tuple

import cv2
from flask import Flask, Response, jsonify, render_template, request

# 说明：
# - 本模块只负责 Web 交互（MJPEG 视频 + 手动框选）
# - 实际帧数据与 bbox 变量由 detect_track_camera 在运行时注入
# - 通过 set_shared_state 在主程序中传入共享引用


class SharedState:
    """保存与 detect_track_camera 共享的状态引用。"""

    def __init__(self) -> None:
        # 这几个成员会在运行时被赋值为『可变对象的引用』，例如：
        # state.latest_ir_frame_ref = {"frame": None}
        self.latest_ir_frame_ref = None  # type: ignore
        self.latest_tv_frame_ref = None  # type: ignore
        self.manual_bbox_ir_ref = None  # type: ignore
        self.manual_bbox_tv_ref = None  # type: ignore


_shared_state = SharedState()


def set_shared_state(latest_ir_frame_ref,
                     latest_tv_frame_ref,
                     manual_bbox_ir_ref,
                     manual_bbox_tv_ref) -> None:
    """由 detect_track_camera 在启动时调用，用于注入共享变量引用。

    latest_*_frame_ref / manual_bbox_*_ref 建议是 dict 或类似可变对象，
    例如： latest_ir_frame_ref = {"frame": None}
           manual_bbox_tv_ref   = {"bbox": None}
    这样 Flask 线程与主推理线程之间共享的是同一个对象。
    """

    _shared_state.latest_ir_frame_ref = latest_ir_frame_ref
    _shared_state.latest_tv_frame_ref = latest_tv_frame_ref
    _shared_state.manual_bbox_ir_ref = manual_bbox_ir_ref
    _shared_state.manual_bbox_tv_ref = manual_bbox_tv_ref


def _gen_mjpeg(frame_key: str):
    """根据 key (ir/tv) 生成 MJPEG 流。"""
    if frame_key == "ir":
        ref = _shared_state.latest_ir_frame_ref
    else:
        ref = _shared_state.latest_tv_frame_ref

    # 简单的保护，防止共享状态未初始化
    if ref is None:
        while True:
            # 还没注入时输出全黑图，避免阻塞
            blank = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get() * 0)
            ok, enc = cv2.imencode(".jpg", blank)
            if not ok:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")

    while True:
        frame = ref.get("frame")  # type: ignore[union-attr]
        if frame is None:
            # 没有最新帧时发送一张空白图，避免浏览器卡死
            blank = 255 * (cv2.UMat(480, 640, cv2.CV_8UC3).get() * 0)
            ok, enc = cv2.imencode(".jpg", blank)
            if not ok:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")
            continue

        ok, enc = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + enc.tobytes() + b"\r\n")


def create_app() -> Flask:
    """创建并返回 Flask 应用实例。

    detect_track_camera 应在启动前调用 set_shared_state 注入共享变量。
    模板目录固定为项目根目录下的 `templates`。
    """

    import os

    # 以当前文件所在目录的上一级作为项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_dir = os.path.join(base_dir, "templates")

    app = Flask(__name__, template_folder=template_dir)

    @app.route("/video_ir")
    def video_ir() -> Response:
        return Response(_gen_mjpeg("ir"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/video_tv")
    def video_tv() -> Response:
        return Response(_gen_mjpeg("tv"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/manual_ir", methods=["POST"])
    def manual_ir():
        data = request.get_json(silent=True) or {}
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 0))
        h = int(data.get("h", 0))
        if _shared_state.manual_bbox_ir_ref is not None:
            _shared_state.manual_bbox_ir_ref["bbox"] = (x, y, w, h)
        return jsonify({"status": "ok", "mode": "IR", "bbox": [x, y, w, h]})

    @app.route("/manual_tv", methods=["POST"])
    def manual_tv():
        data = request.get_json(silent=True) or {}
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 0))
        h = int(data.get("h", 0))
        if _shared_state.manual_bbox_tv_ref is not None:
            _shared_state.manual_bbox_tv_ref["bbox"] = (x, y, w, h)
        return jsonify({"status": "ok", "mode": "TV", "bbox": [x, y, w, h]})

    @app.route("/manual")
    def manual_page():
        # 简单页面：左右各一路视频，可在 TV 上拖拽框选
        return render_template("manual_control.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def run_app_in_thread(host: str = "0.0.0.0", port: int = 5000) -> threading.Thread:
    """以守护线程方式启动 Flask 服务器，供主程序调用。

    典型用法：
        from web.manual_control import set_shared_state, create_app, run_app_in_thread
        set_shared_state(...)
        app = create_app()
        run_app_in_thread()
    """

    app = create_app()

    def _run():
        app.run(host=host, port=port, threaded=True, debug=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
