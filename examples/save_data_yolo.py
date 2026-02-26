# *******************************************************************************
# Integrated: Color + Depth + PointCloud + YOLO OBB save
# (Threaded, pair guaranteed)
# *******************************************************************************

import os
import cv2
import numpy as np
import time
import threading
import queue
import signal
import sys

from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO

# ===============================
# YOLO OBB 모델 로드
# ===============================
#MODEL_PATH = "/home/myung/workspace/paper_best_all.pt"
#
#MODEL_PATH = "/home/myung/workspace/paper_test/yolo11_models/yolo11n.pt"
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



def save_worker():
    global save_count, save_last_time

    # PointCloud filter (1회 생성)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    # 폴더 미리 생성
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
                # 원본 이미지 저장
                cv2.imwrite(
                    f"color_images/color_{index}_{timestamp}.jpg",
                    img
                )

                # ------------------------------
                # YOLO OBB inference
                # ------------------------------
                results = model.predict(
                    img,
                    conf=0.3,
                    imgsz=640,
                    device=0,
                    verbose=False
                )

                r = results[0]

                # ------------------------------
                # 시각화 이미지 저장
                # ------------------------------
                vis = r.plot()
                cv2.imwrite(
                    f"pred_images/pred_{index}_{timestamp}.jpg",
                    vis
                )

                # ------------------------------
                # OBB label 저장
                # format:
                # class x1 y1 x2 y2 x3 y3 x4 y4
                # ------------------------------
                if r.obb is not None:
                    with open(f"labels/label_{index}.txt", "w") as f:
                        for cls, pts in zip(r.obb.cls, r.obb.xyxyxyxy):
                            pts = pts.cpu().numpy().reshape(-1)
                            line = f"{int(cls)} " + " ".join(map(str, pts))
                            f.write(line + "\n")

        # ==================================================
        # 2. POINT CLOUD
        # ==================================================

        if index % 10 == 0:
            pc_frame = point_cloud_filter.process(frames)
            if pc_frame:
                save_point_cloud_to_ply(
                    f"point_clouds/cloud_{index}.ply",
                    pc_frame
                )
                print(f"point_clouds/cloud_{index}.ply")

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
    cv2.setNumThreads(4)  # Jetson OpenCV 최적화

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
    thread = threading.Thread(
        target=save_worker,
        daemon=True
    )
    thread.start()

    index = 0
    frame_count = 0
    last_time = time.time()

    print("Start capture (Threaded saving + YOLO)")
    last_capture_time = 0

    while True:
        try:
            now = time.time()
            if now - last_capture_time < FRAME_INTERVAL:
                time.sleep(0.001)   # CPU 100% 방지
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
            # Queue push (빠름)
            # ===============================
            try:
                save_queue.put_nowait(
                    (color_frame, depth_frame, frames, has_color_sensor, index)
                )
                index += 1
            except queue.Full:
                pass  # 저장 밀리면 drop

        except KeyboardInterrupt:
            break

    # ===============================
    # Shutdown
    # ===============================
    running = False
    save_queue.put(None)
    thread.join()

    pipeline.stop()
    print("Finished")


if __name__ == "__main__":
    main()
