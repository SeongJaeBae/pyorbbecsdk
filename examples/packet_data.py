import os
import sys
import cv2
import time
import socket
import struct
import threading
import numpy as np
from datetime import datetime

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_XCB_GL_INTEGRATION"] = "none"
os.environ["QT_OPENGL"] = "software"

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QTextCursor, QImage
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread

from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO


# =========================================================
# Config
# =========================================================
MODEL_PATH = "/home/swcho/workspace/pyorbbecsdk/examples/obb_best_260109.pt"

SERVER_IP = "192.168.1.142"
SERVER_PORT = 9000
CAMERA_TYPE = "orbbec"

CAPTURE_FPS = 30
CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS

INFER_CONF = 0.3
INFER_IMGSZ = 640
INFER_DEVICE = 0

SEND_POINTCLOUD = True
POINTCLOUD_SEND_EVERY = 10  # pred frame 기준 N회마다 1번

MOTION_ENABLED = True
MOTION_SENSITIVITY = 3000
MOTION_THRESHOLD = 25
BLUR_KERNEL = (21, 21)
MOTION_COOLDOWN_SEC = 0.2


# =========================================================
# Utils
# =========================================================
def cvimg_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    if img_bgr is None:
        return QPixmap()

    if len(img_bgr.shape) == 2:
        h, w = img_bgr.shape
        qimg = QImage(
            img_bgr.data, w, h, img_bgr.strides[0], QImage.Format.Format_Grayscale8
        )
        return QPixmap.fromImage(qimg.copy())

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(
        rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888
    )
    return QPixmap.fromImage(qimg.copy())


def preprocess_for_motion(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL, 0)


def detect_motion(prev_gray: np.ndarray, curr_gray: np.ndarray):
    delta = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    score = cv2.countNonZero(thresh)
    return score, thresh


def depth_frame_to_colormap(depth_frame: DepthFrame) -> np.ndarray | None:
    if depth_frame is None:
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    if data.size != width * height:
        return None

    depth = data.reshape((height, width))
    depth_mm = depth.astype(np.float32) * scale

    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)

    vis_max = np.percentile(depth_mm[valid], 99)
    vis_max = max(vis_max, 1000.0)

    depth_norm = np.clip(depth_mm / vis_max * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_color[~valid] = 0
    return depth_color


def save_point_cloud_to_memory(pc_frame):
    if pc_frame is None:
        return None

    data = pc_frame.get_data()
    if data is None:
        return None

    points_np = np.frombuffer(data, dtype=np.float32)
    if points_np.size == 0:
        return None

    points_np = points_np.reshape(-1, 6)
    vertex_count = points_np.shape[0]

    header = f"""ply
format binary_little_endian 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    header_bytes = header.encode("utf-8")

    xyz = points_np[:, :3]
    rgb = points_np[:, 3:6].astype(np.uint8)

    structured = np.empty(
        vertex_count,
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("r", np.uint8),
            ("g", np.uint8),
            ("b", np.uint8),
        ]
    )
    structured["x"] = xyz[:, 0]
    structured["y"] = xyz[:, 1]
    structured["z"] = xyz[:, 2]
    structured["r"] = rgb[:, 0]
    structured["g"] = rgb[:, 1]
    structured["b"] = rgb[:, 2]

    return header_bytes + structured.tobytes()


# =========================================================
# Sender
# =========================================================
class PersistentSender:
    def __init__(self, ip, port, log_callback=None):
        self.ip = ip
        self.port = port
        self.sock = None
        self.lock = threading.Lock()
        self.log_callback = log_callback

    def log(self, text: str):
        if self.log_callback:
            self.log_callback(text)
        else:
            print(text)

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(20)
        self.sock.connect((self.ip, self.port))
        self.log(f"[TCP] Connected to {self.ip}:{self.port}")

    def send(self, buffer_bytes: bytes, filename: str, file_type: str, capture_id: str):
        with self.lock:
            try:
                if self.sock is None:
                    self._connect()

                header = f"{CAMERA_TYPE}|{file_type}|{capture_id}|{filename}"
                header_bytes = header.encode()

                self.sock.sendall(struct.pack("!I", len(header_bytes)))
                self.sock.sendall(header_bytes)
                self.sock.sendall(struct.pack("!Q", len(buffer_bytes)))
                self.sock.sendall(buffer_bytes)

                _ = self.sock.recv(4096)
                self.log(f"[SEND] {filename} ({len(buffer_bytes)} bytes)")

            except Exception as e:
                self.log(f"[SEND ERROR] reconnecting: {e}")
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# =========================================================
# Shared frame buffer (latest-frame only)
# =========================================================
class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.version = 0
        self.frame_data = None

    def update(self, data: dict):
        with self.lock:
            self.version += 1
            data["version"] = self.version
            self.frame_data = data

    def get(self):
        with self.lock:
            if self.frame_data is None:
                return None
            return dict(self.frame_data)


# =========================================================
# Signals
# =========================================================
class WorkerSignals(QObject):
    pred_image = pyqtSignal(QPixmap)
    depth_image = pyqtSignal(QPixmap)
    log = pyqtSignal(str)
    status = pyqtSignal(str)
    finished = pyqtSignal()


# =========================================================
# Main Worker
# =========================================================
class CaptureWorker(QThread):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True

        self.pipeline = None
        self.model = YOLO(MODEL_PATH)
        self.sender = PersistentSender(SERVER_IP, SERVER_PORT, self.emit_log)

        self.latest_buffer = LatestFrameBuffer()
        self.capture_thread = None
        self.infer_thread = None

        self.prev_gray = None
        self.last_motion_time = 0.0

        self.capture_fps_count = 0
        self.capture_fps_last = time.time()

        self.infer_fps_count = 0
        self.infer_fps_last = time.time()

        self.sent_pred_count = 0

    def emit_log(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.signals.log.emit(f"[{now}] {text}")

    def stop(self):
        self.running = False

    def run(self):
        try:
            self.pipeline = Pipeline()
            config = Config()
            has_color_sensor = False

            try:
                profile_list = self.pipeline.get_stream_profile_list(
                    OBSensorType.COLOR_SENSOR
                )
                color_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                has_color_sensor = True
                self.emit_log("[CAM] Color sensor enabled")
            except Exception as e:
                self.emit_log(f"[CAM] Color sensor not available: {e}")

            depth_profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            self.emit_log("[CAM] Depth sensor enabled")

            self.pipeline.enable_frame_sync()
            self.pipeline.start(config)
            self.emit_log("[CAM] Pipeline started")

            self.capture_thread = threading.Thread(
                target=self.capture_loop, daemon=True
            )
            self.infer_thread = threading.Thread(
                target=self.infer_loop, daemon=True
            )

            self.capture_thread.start()
            self.infer_thread.start()

            self.signals.status.emit("Running")

            while self.running:
                time.sleep(0.1)

        except Exception as e:
            self.emit_log(f"[FATAL] {e}")

        finally:
            self.cleanup()
            self.signals.finished.emit()

    def capture_loop(self):
        last_capture_time = 0.0

        while self.running:
            try:
                now = time.time()
                if now - last_capture_time < CAPTURE_INTERVAL:
                    time.sleep(0.001)
                    continue
                last_capture_time = now

                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if depth_frame is None or color_frame is None:
                    continue

                color_img = frame_to_bgr_image(color_frame)
                if color_img is None:
                    continue

                depth_vis = depth_frame_to_colormap(depth_frame)
                if depth_vis is not None:
                    self.signals.depth_image.emit(cvimg_to_qpixmap(depth_vis))

                motion_detected = True
                motion_score = 0

                if MOTION_ENABLED:
                    curr_gray = preprocess_for_motion(color_img)

                    if self.prev_gray is None:
                        self.prev_gray = curr_gray
                        continue

                    motion_score, _ = detect_motion(self.prev_gray, curr_gray)
                    self.prev_gray = curr_gray

                    if motion_score > MOTION_SENSITIVITY and \
                       (time.time() - self.last_motion_time) >= MOTION_COOLDOWN_SEC:
                        motion_detected = True
                        self.last_motion_time = time.time()
                    else:
                        motion_detected = False

                self.capture_fps_count += 1
                now = time.time()
                if now - self.capture_fps_last >= 1.0:
                    self.emit_log(f"[CAPTURE FPS] {self.capture_fps_count}")
                    self.capture_fps_count = 0
                    self.capture_fps_last = now

                self.signals.status.emit(
                    f"Running | Motion: {motion_score} | Trigger: {motion_detected}"
                )

                if not motion_detected:
                    continue

                capture_id = datetime.now().isoformat(timespec="milliseconds")

                # 핵심: queue에 넣는게 아니라 최신 프레임만 덮어씀
                self.latest_buffer.update({
                    "capture_id": capture_id,
                    "color_img": color_img.copy(),
                    "depth_vis": depth_vis.copy() if depth_vis is not None else None,
                    "color_frame": color_frame,
                    "depth_frame": depth_frame,
                    "frames": frames,
                })

            except Exception as e:
                self.emit_log(f"[CAPTURE ERROR] {e}")
                time.sleep(0.01)

    def infer_loop(self):
        last_processed_version = -1
        point_cloud_filter = PointCloudFilter()
        point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

        while self.running:
            try:
                item = self.latest_buffer.get()
                if item is None:
                    time.sleep(0.005)
                    continue

                version = item["version"]
                if version == last_processed_version:
                    time.sleep(0.005)
                    continue

                last_processed_version = version

                capture_id = item["capture_id"]
                color_img = item["color_img"]
                depth_vis = item["depth_vis"]
                frames = item["frames"]

                # 1) 원본 color 전송
                ok, encoded_color = cv2.imencode(".jpg", color_img)
                if ok:
                    self.sender.send(
                        encoded_color.tobytes(),
                        f"color_{capture_id}.jpg",
                        "color",
                        capture_id
                    )

                # 2) 최신 프레임으로 바로 추론
                results = self.model.predict(
                    color_img,
                    conf=INFER_CONF,
                    imgsz=INFER_IMGSZ,
                    device=INFER_DEVICE,
                    verbose=False
                )
                r = results[0]
                pred_vis = r.plot()

                # 3) GUI 갱신
                self.signals.pred_image.emit(cvimg_to_qpixmap(pred_vis))

                # 4) pred 전송
                ok, encoded_pred = cv2.imencode(".jpg", pred_vis)
                if ok:
                    self.sender.send(
                        encoded_pred.tobytes(),
                        f"pred_{capture_id}.jpg",
                        "prediction",
                        capture_id
                    )

                # 5) OBB label 전송
                if r.obb is not None and len(r.obb.cls) > 0:
                    label_lines = []
                    for cls, pts in zip(r.obb.cls, r.obb.xyxyxyxy):
                        pts = pts.cpu().numpy().reshape(-1)
                        line = f"{int(cls)} " + " ".join(map(str, pts))
                        label_lines.append(line)

                    label_str = "\n".join(label_lines) + "\n"
                    self.sender.send(
                        label_str.encode(),
                        f"label_{capture_id}.txt",
                        "label",
                        capture_id
                    )

                # 6) depth 전송
                if depth_vis is not None:
                    ok, encoded_depth = cv2.imencode(".jpg", depth_vis)
                    if ok:
                        self.sender.send(
                            encoded_depth.tobytes(),
                            f"depth_{capture_id}.jpg",
                            "depth",
                            capture_id
                        )

                # 7) pointcloud 전송
                self.sent_pred_count += 1
                if SEND_POINTCLOUD and (self.sent_pred_count % POINTCLOUD_SEND_EVERY == 0):
                    pc_frame = point_cloud_filter.process(frames)
                    if pc_frame is not None:
                        ply_bytes = save_point_cloud_to_memory(pc_frame)
                        if ply_bytes is not None:
                            self.sender.send(
                                ply_bytes,
                                f"cloud_{capture_id}.ply",
                                "pointcloud",
                                capture_id
                            )

                self.infer_fps_count += 1
                now = time.time()
                if now - self.infer_fps_last >= 1.0:
                    self.emit_log(f"[INFER FPS] {self.infer_fps_count}")
                    self.infer_fps_count = 0
                    self.infer_fps_last = now

            except Exception as e:
                self.emit_log(f"[INFER ERROR] {e}")
                time.sleep(0.01)

    def cleanup(self):
        self.emit_log("[SAFE EXIT] Releasing resources...")
        self.sender.close()

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.infer_thread is not None and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=1.0)

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.signals.status.emit("Stopped")


# =========================================================
# GUI
# =========================================================
class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orbbec Pred / Depth Viewer (Latest Frame)")

        self.left_title = QLabel("Prediction")
        self.right_title = QLabel("Depth")

        for title in [self.left_title, self.right_title]:
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setMaximumHeight(36)
            title.setStyleSheet("""
                background-color: #f0f0f0;
                color: black;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #cccccc;
            """)

        self.left_label = QLabel("Pred")
        self.right_label = QLabel("Depth")
        self.status = QLabel("Ready")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.left_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for lbl in [self.left_label, self.right_label]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #111111; color: white;")

        self.status.setMaximumHeight(32)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-size: 15px; padding: 4px;")

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.start_btn.clicked.connect(self.start_worker)
        self.stop_btn.clicked.connect(self.stop_worker)

        self.log_box.setMaximumHeight(220)
        self.log_box.setStyleSheet("background-color: #0b0b0b; color: #dddddd;")

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(0)
        left_panel.addWidget(self.left_title, 0)
        left_panel.addWidget(self.left_label, 1)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)
        right_panel.addWidget(self.right_title, 0)
        right_panel.addWidget(self.right_label, 1)

        layout_img = QHBoxLayout()
        layout_img.setContentsMargins(0, 0, 0, 0)
        layout_img.setSpacing(6)
        layout_img.addLayout(left_panel, 1)
        layout_img.addLayout(right_panel, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addLayout(layout_img, 4)
        layout.addWidget(self.status, 0)
        layout.addLayout(btn_layout, 0)
        layout.addWidget(self.log_box, 1)

        self.setLayout(layout)

        self.worker = None
        self.current_pred_pixmap = None
        self.current_depth_pixmap = None

    def append_log(self, text: str):
        if not text:
            return
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.insertPlainText(text + "\n")
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)

    def set_pixmap_fit(self, label, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        scaled = pixmap.scaled(
            max(1, label.width()),
            max(1, label.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def update_pred(self, pixmap: QPixmap):
        self.current_pred_pixmap = pixmap
        self.set_pixmap_fit(self.left_label, pixmap)

    def update_depth(self, pixmap: QPixmap):
        self.current_depth_pixmap = pixmap
        self.set_pixmap_fit(self.right_label, pixmap)

    def start_worker(self):
        self.worker = CaptureWorker()
        self.worker.signals.pred_image.connect(self.update_pred)
        self.worker.signals.depth_image.connect(self.update_depth)
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.status.connect(self.status.setText)
        self.worker.signals.finished.connect(self.on_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Starting...")
        self.log_box.clear()

        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.status.setText("Stopping...")

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopped")
        self.append_log("[GUI] Worker finished")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pred_pixmap is not None:
            self.set_pixmap_fit(self.left_label, self.current_pred_pixmap)
        if self.current_depth_pixmap is not None:
            self.set_pixmap_fit(self.right_label, self.current_depth_pixmap)

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# =========================================================
# main
# =========================================================
def main():
    app = QApplication(sys.argv)
    viewer = Viewer()
    viewer.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
