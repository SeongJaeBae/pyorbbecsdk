# -*- coding: utf-8 -*-
import os
import cv2

from pyorbbecsdk import *
from utils import frame_to_bgr_image


SAVE_VIDEO_PATH = "output.mp4"
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


def main():
    pipeline = Pipeline()
    config = Config()
    writer = None
    frame_index = 0

    try:
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profile_list is None:
            print("[ERROR] No color sensor")
            return

        color_profile = get_color_profile_1280x720(color_profile_list)
        if color_profile is None:
            print("[ERROR] Failed to get color profile")
            return

        config.enable_stream(color_profile)

        print(
            f"[INFO] Using: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()}fps"
        )

        pipeline.start(config)

        for _ in range(10):
            pipeline.wait_for_frames(100)

        # mp4 저장용 codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

        if not writer.isOpened():
            print("[ERROR] Failed to open VideoWriter")
            return

        print("[INFO] Recording started... (Ctrl+C to stop)")

        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            image = frame_to_bgr_image(color_frame)
            if image is None:
                continue

            # 혹시 실제 해상도가 다르면 맞춰줌
            if image.shape[1] != WIDTH or image.shape[0] != HEIGHT:
                image = cv2.resize(image, (WIDTH, HEIGHT))

            writer.write(image)
            frame_index += 1

            print(f"\r[INFO] Recorded frames: {frame_index}", end="")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    except OBError as e:
        print(f"\n[OBError] {e}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
    finally:
        if writer is not None:
            writer.release()
        try:
            pipeline.stop()
        except Exception:
            pass
        print(f"[INFO] Video saved to: {SAVE_VIDEO_PATH}")


if __name__ == "__main__":
    main()