import os
import cv2
import numpy as np
import time
import threading
import queue
import signal
import sys
import socket
import struct
import uuid
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO

# ===============================
# YOLO OBB 모델 로드
# ===============================
MODEL_PATH = "/home/myung/workspace/obb_best_260109.pt"
model = YOLO(MODEL_PATH)

# ===============================
# Queue + Worker
# ===============================
save_queue = queue.Queue(maxsize=50)
running = True

save_count = 0
save_last_time = time.time()
pipeline = None

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

SERVER_IP = '192.168.1.154'
SERVER_PORT = 9000
CAMERA_TYPE = "orbbec"


# ===============================
# Persistent TCP Sender
# ===============================
class PersistentSender:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.lock = threading.Lock()

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(20)
        self.sock.connect((self.ip, self.port))
        print(f"Connected to {self.ip}:{self.port}")

    def send(self, buffer_bytes, filename, file_type, capture_id):
        with self.lock:
            try:
                if self.sock is None:
                    self._connect()

                header = f"{CAMERA_TYPE}|{file_type}|{capture_id}|{filename}"
                header_bytes = header.encode()

                self.sock.sendall(struct.pack('!I', len(header_bytes)))
                self.sock.sendall(header_bytes)
                self.sock.sendall(struct.pack('!Q', len(buffer_bytes)))
                self.sock.sendall(buffer_bytes)

                response = self.sock.recv(4096)
                print(f"sent: {filename} ({len(buffer_bytes)} bytes)")

            except Exception as e:
                print(f"SEND ERROR, reconnecting: {e}")
                try:
                    self.sock.close()
                except:
                    pass
                self.sock = None

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass


sender = PersistentSender(SERVER_IP, SERVER_PORT)


# ===============================
# Point Cloud → PLY bytes
# ===============================
def save_point_cloud_to_memory(pc_frame):
    if pc_frame is None:
        return None

    data = pc_frame.get_data()
    if data is None:
        return None

    points_np = np.frombuffer(data, dtype=np.float32)
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
    header_bytes = header.encode('utf-8')

    xyz = points_np[:, :3]
    rgb = points_np[:, 3:6].astype(np.uint8)

    structured = np.empty(
        vertex_count,
        dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('r', np.uint8),
            ('g', np.uint8),
            ('b', np.uint8),
        ]
    )
    structured['x'] = xyz[:, 0]
    structured['y'] = xyz[:, 1]
    structured['z'] = xyz[:, 2]
    structured['r'] = rgb[:, 0]
    structured['g'] = rgb[:, 1]
    structured['b'] = rgb[:, 2]

    return header_bytes + structured.tobytes()


# ===============================
# Save Worker Thread
# ===============================
def save_worker():
    global save_count, save_last_time

    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    os.makedirs("color_images", exist_ok=True)
    os.makedirs("pred_images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    os.makedirs("point_clouds", exist_ok=True)

    while running:
        item = save_queue.get()
        if item is None:
            break

        color_frame, depth_frame, frames, has_color, index = item

        # ==================================================
        # 1. COLOR + YOLO
        # ==================================================
        if color_frame:
            timestamp = color_frame.get_timestamp()
            img = frame_to_bgr_image(color_frame)

            if img is not None:
                filename = f"color_{index}_{timestamp}.jpg"

                success, encoded_img = cv2.imencode(".jpg", img)
                if success:
                    sender.send(
                        encoded_img.tobytes(), filename, "color", index
                    )

                results = model.predict(
                    img, conf=0.3, imgsz=640, device=0, verbose=False
                )
                r = results[0]

                vis = r.plot()
                
                success, encoded_vis = cv2.imencode(".jpg", vis)
                if success:
                    sender.send(
                        encoded_vis.tobytes(),
                        f"pred_{index}_{timestamp}.jpg",
                        "prediction",
                        index
                    )

                if r.obb is not None:
                    label_str = ""
                    for cls, pts in zip(r.obb.cls, r.obb.xyxyxyxy):
                        pts = pts.cpu().numpy().reshape(-1)
                        line = f"{int(cls)} " + " ".join(map(str, pts))
                        label_str += line + "\n"

                    sender.send(
                        label_str.encode(),
                        f"label_{index}.txt",
                        "label",
                        index
                    )

        # ==================================================
        # 2. POINT CLOUD
        # ==================================================
        if save_count % 10 == 0:
            pc_frame = point_cloud_filter.process(frames)
            if pc_frame:
                ply_bytes = save_point_cloud_to_memory(pc_frame)
                sender.send(
                    ply_bytes, f"cloud_{index}.ply", "pointcloud", index
                )

        # ==================================================
        # SAVE FPS
        # ==================================================
        save_count += 1
        now = time.time()
        if now - save_last_time >= 1:
            print(f"[SAVE FPS] {save_count}")
            save_count = 0
            save_last_time = now

        save_queue.task_done()


def cleanup(signum=None, frame=None):
    global running, pipeline
    print("\n[SAFE EXIT] Releasing camera...")
    running = False
    sender.close()
    try:
        pipeline.stop()
    except:
        pass
    return


# ===============================
# main
# ===============================
def main():
    global running, pipeline
    cv2.setNumThreads(4)

    pipeline = Pipeline()
    config = Config()
    has_color_sensor = False

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ---------- COLOR ----------
    try:
        profile_list = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        has_color_sensor = True
    except Exception as e:
        print("Color sensor not available:", e)

    # ---------- DEPTH ----------
    depth_profile_list = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.enable_frame_sync()
    pipeline.start(config)

    # ===============================
    # Start save thread
    # ===============================
    thread = threading.Thread(target=save_worker, daemon=True)
    thread.start()

    
    frame_count = 0
    last_time = time.time()
    last_capture_time = 0

    print("Start capture (Threaded saving + YOLO)")

    while True:
        try:
            now = time.time()
            if now - last_capture_time < FRAME_INTERVAL:
                time.sleep(0.001)
                continue

            last_capture_time = now

            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if depth_frame is None:
                continue
            if has_color_sensor and color_frame is None:
                continue

            # ===============================
            # CAPTURE FPS
            # ===============================
            frame_count += 1
            now = time.time()
            if now - last_time >= 1:
                print(f"[CAPTURE FPS] {frame_count}")
                frame_count = 0
                last_time = now

            # ===============================
            # Queue push
            # ===============================
            index = str(uuid.uuid4())
            try:
                save_queue.put_nowait(
                    (color_frame, depth_frame, frames, has_color_sensor, index)
                )
            except queue.Full:
                pass

        except KeyboardInterrupt:
            break

    # ===============================
    # Shutdown
    # ===============================
    running = False
    save_queue.put(None)
    thread.join()
    sender.close()
    pipeline.stop()
    print("Finished")


if __name__ == "__main__":
    main()
