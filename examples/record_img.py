# -*- coding: utf-8 -*-
import os
import cv2

from pyorbbecsdk import *
from utils import frame_to_bgr_image


# =========================
# 설정
# =========================
SAVE_DIR = "saved_images"
WIDTH = 1280
HEIGHT = 720
FPS = 30


def get_color_profile_1280x720(profile_list):
    if profile_list is None:
        return None

    try:
        count = profile_list.get_count()
        for i in range(count):
            profile = profile_list.get_video_stream_profile(i)
            if profile is None:
                continue

            if profile.get_width() == WIDTH and profile.get_height() == HEIGHT:
                return profile
    except Exception as e:
        print(f"[WARN] profile search failed: {e}")

    print("[WARN] Using default profile")
    return profile_list.get_default_video_stream_profile()


def save_color_frame(frame: ColorFrame, index: int):
    if frame is None:
        return

    image = frame_to_bgr_image(frame)
    if image is None:
        print("[WARN] convert fail")
        return

    filename = os.path.join(SAVE_DIR, f"image_{index:06d}.jpg")
    cv2.imwrite(filename, image)


def main():
    pipeline = Pipeline()
    config = Config()

    os.makedirs(SAVE_DIR, exist_ok=True)

    frame_index = 0

    try:
        # 컬러 스트림 설정
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profile_list is None:
            print("[ERROR] No color sensor")
            return

        color_profile = get_color_profile_1280x720(color_profile_list)
        config.enable_stream(color_profile)

        print(
            f"[INFO] Using: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps"
        )

        pipeline.start(config)

        # 센서 안정화
        for _ in range(10):
            pipeline.wait_for_frames(100)

        print("[INFO] Saving ALL frames... (Ctrl+C to stop)")

        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            frame_index += 1
            save_color_frame(color_frame, frame_index)

            # 로그 (너무 많으면 주석)
            print(f"Saved: image_{frame_index:06d}.jpg")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")
    except OBError as e:
        print(e)
    finally:
        pipeline.stop()
        print("[INFO] Pipeline stopped")


if __name__ == "__main__":
    main()