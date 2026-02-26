# *******************************************************************************
# Integrated: Color + Depth + PointCloud save (Threaded, pair guaranteed)
# *******************************************************************************

import os
import cv2
import numpy as np
import time
import threading
import queue

from pyorbbecsdk import *
from utils import frame_to_bgr_image

from ultralytics import YOLO

# ===============================
# YOLO OBB 모델 로드
# ===============================
MODEL_PATH = "/home/myung/workspace/paper_best_all.pt"  # ← 본인 모델 경로
model = YOLO(MODEL_PATH)

# model.to("cuda").half()     # FP16 → 1.7~2배 빨라짐

# ===============================
# Queue + Worker
# ===============================
save_queue = queue.Queue(maxsize=50)

running = True

save_count = 0

save_last_time = time.time()


def save_worker():
    global save_count, save_last_time

    point_cloud_filter = PointCloudFilter()  # 1번만 생성 (속도 ↑↑)
    fmt = OBFormat.RGB_POINT
    point_cloud_filter.set_create_point_format(fmt)

    while running:
        item = save_queue.get()
        if item is None:
            break

        color_frame, depth_frame, frames, has_color, index = item

        # ---------- color ----------
        if color_frame:
            timestamp = color_frame.get_timestamp()
            img = frame_to_bgr_image(color_frame)
            if img is not None:
                os.makedirs("color_images", exist_ok=True)
                cv2.imwrite(f"color_images/color_{index}_{timestamp}.jpg", img)

        # ---------- pointcloud ----------
        pc_frame = point_cloud_filter.process(frames)
        if pc_frame:
            os.makedirs("point_clouds", exist_ok=True)
            save_point_cloud_to_ply(f"point_clouds/cloud_{index}.ply", pc_frame)

        # ===============================
        # 실제 저장 FPS
        # ===============================
        save_count += 1
        now = time.time()
        if now - save_last_time >= 1:
            print(f"[SAVE FPS] {save_count}")
            save_count = 0
            save_last_time = now

        save_queue.task_done()


# ===============================
# main
# ===============================
def main():
    global running

    cv2.setNumThreads(4)  # Jetson 최적화

    pipeline = Pipeline()
    config = Config()

    has_color_sensor = False

    # ---------- color ----------
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        has_color_sensor = True
    except:
        pass

    # ---------- depth ----------
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.enable_frame_sync()
    pipeline.start(config)

    # ===============================
    # start save thread
    # ===============================
    thread = threading.Thread(target=save_worker, daemon=True)
    thread.start()

    index = 0
    frame_count = 0
    last_time = time.time()

    print("Start capture (Threaded saving)")

    while True:
        try:
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
            # FPS
            # ===============================
            frame_count += 1
            now = time.time()
            if now - last_time >= 1:
                print(f"[FPS] {frame_count}")
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

    running = False
    save_queue.put(None)
    thread.join()

    pipeline.stop()
    print("Finished")


if __name__ == "__main__":
    main()
