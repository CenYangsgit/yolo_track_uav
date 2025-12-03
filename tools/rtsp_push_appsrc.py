import cv2
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GObject, GstRtspServer
import numpy as np
import threading
import time
import socket

"""
rtsp_push_appsrc.py (Final Fix for Dual Stream)
修订说明:
1. 实现全局单例模式 (Singleton)：多个 RTSPFramePusher 实例共享同一个 RTSP Server 和 MainLoop。
   这样可以在同一个 8554 端口上同时挂载 /ir 和 /tv，解决端口冲突问题。
2. 保留了之前修复的 crash bug 和硬件编码优化。
"""

# 全局变量，用于在不同实例间共享服务器和事件循环
_GLOBAL_RTSP_SERVER = None
_GLOBAL_MAIN_LOOP = None
_GLOBAL_LOOP_THREAD = None
_GLOBAL_LOCK = threading.Lock()

class RTSPFramePusher:
    def __init__(self, width, height, fps=25, mount="/test", encoder="x264", bitrate=2048):
        global _GLOBAL_RTSP_SERVER, _GLOBAL_MAIN_LOOP

        self.width = width
        self.height = height
        self.fps = fps
        self.mount = mount
        self.encoder = encoder
        self.bitrate = bitrate
        self.latest_frame = None
        self.lock = threading.Lock()
        self.ts_counter = 0

        Gst.init(None)

        # -----------------------------------------------------------
        # 关键修改：使用全局共享的 Server，防止端口冲突
        # -----------------------------------------------------------
        with _GLOBAL_LOCK:
            if _GLOBAL_RTSP_SERVER is None:
                _GLOBAL_RTSP_SERVER = GstRtspServer.RTSPServer.new()
                _GLOBAL_RTSP_SERVER.set_service("8554")
                try:
                    _GLOBAL_RTSP_SERVER.set_address("0.0.0.0")
                except Exception:
                    pass
                _GLOBAL_RTSP_SERVER.attach(None)
                print(f"[rtsp] Gloabl Server created on port 8554 (bind 0.0.0.0)")
            
            # 复用已有的服务器实例
            self.server = _GLOBAL_RTSP_SERVER

            if _GLOBAL_MAIN_LOOP is None:
                _GLOBAL_MAIN_LOOP = GObject.MainLoop()

        # 为当前挂载点创建工厂
        self.factory = GstRtspServer.RTSPMediaFactory.new()
        self.factory.set_shared(True)
        pipeline = self._build_pipeline()
        self.factory.set_launch(pipeline)
        
        # 将工厂挂载到共享服务器上
        self.server.get_mount_points().add_factory(self.mount, self.factory)

        self.feed = False

    def _build_pipeline(self):
        if self.encoder == "x264":
            enc = f"x264enc tune=zerolatency bitrate={self.bitrate} speed-preset=veryfast key-int-max={self.fps}"
        elif self.encoder == "v4l2h264enc":
            enc = f"v4l2h264enc extra-controls=\"controls,video_bitrate={self.bitrate*1000}\""
        elif self.encoder == "mpph264enc":
            # 硬件编码优化
            enc = f"mpph264enc profile=66 gop={self.fps} bps={self.bitrate*1000}"
        else:
            enc = f"x264enc tune=zerolatency bitrate={self.bitrate} speed-preset=veryfast key-int-max={self.fps}"

        pipeline = (
            f"( appsrc name=mysrc is-live=true format=GST_FORMAT_TIME do-timestamp=true ! "
            f"videoconvert ! video/x-raw,format=I420 ! {enc} ! "
            f"h264parse config-interval=1 ! rtph264pay pt=96 name=pay0 )"
        )
        return pipeline

    def start(self):
        global _GLOBAL_LOOP_THREAD, _GLOBAL_MAIN_LOOP
        self.feed = True
        
        # 启动全局事件循环（如果还没启动的话）
        with _GLOBAL_LOCK:
            if _GLOBAL_LOOP_THREAD is None:
                def run_loop():
                    if _GLOBAL_MAIN_LOOP:
                        _GLOBAL_MAIN_LOOP.run()
                t = threading.Thread(target=run_loop, daemon=True)
                t.start()
                _GLOBAL_LOOP_THREAD = t
                print("[rtsp] Global MainLoop started.")

    def stop(self):
        self.feed = False
        # 注意：在共享模式下，单个实例 stop 不应关闭全局 Loop，
        # 除非确定所有流都关闭了。这里简单处理，不退出 Loop。
        pass

    def push(self, frame_bgr):
        if frame_bgr is None:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        with self.lock:
            self.latest_frame = frame_bgr.copy()

    def bind_appsrc(self, media):
        element = media.get_element()
        appsrc = element.get_by_name("mysrc")
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1"
        )
        appsrc.set_property("caps", caps)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("block", True)
        appsrc.connect("need-data", self._on_need_data)

    def _on_need_data(self, src, length):
        if not self.feed:
            return
        with self.lock:
            fr = self.latest_frame
        if fr is None:
            fr = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        data = fr.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        duration = int((1 / max(1, self.fps)) * 1e9)
        self.ts_counter += duration
        
        # 使用属性赋值，避免 AttributeError
        buf.duration = duration
        buf.pts = self.ts_counter
        buf.dts = self.ts_counter

        src.emit("push-buffer", buf)

def run_server(width, height, fps, mount, encoder, bitrate):
    pusher = RTSPFramePusher(width, height, fps, mount, encoder, bitrate)
    pusher.factory.connect("media-configure", lambda f, m: pusher.bind_appsrc(m))
    pusher.start()
    
    # 获取 IP 仅用于打印提示
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"
        
    print(f"[rtsp] Stream ready at rtsp://{ip}:8554{mount} (Shared Server) encoder={encoder}")
    return pusher

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--mount", type=str, default="/test")
    parser.add_argument("--encoder", type=str, choices=["x264","v4l2h264enc","mpph264enc"], default="x264")
    parser.add_argument("--bitrate", type=int, default=2048)
    args = parser.parse_args()

    pusher = run_server(args.width, args.height, args.fps, args.mount, args.encoder, args.bitrate)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pusher.stop()