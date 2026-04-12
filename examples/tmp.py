import time
import cv2
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from pyorbbecsdk import *
from utils import frame_to_bgr_image


class OrbbecCamera:
    def __init__(self, warmup_frames=10, timeout_ms=1000):
        self.pipeline = None
        self.config = None
        self.started = False
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms

    def open(self):
        self.pipeline = Pipeline()
        self.config = Config()

        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)

        self.pipeline.start(self.config)
        self.started = True

        for _ in range(self.warmup_frames):
            try:
                self.pipeline.wait_for_frames(self.timeout_ms)
            except Exception:
                pass

        print("Orbbec camera opened.")

    def close(self):
        if self.pipeline and self.started:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.started = False
        print("Orbbec camera closed.")

    def grab_frame(self):
        print("[1] wait_for_frames")
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            print("[2] no frames")
            return None

        print("[3] get_color_frame")
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("[4] no color frame")
            return None

        print("[5] frame_to_bgr_image")
        img = frame_to_bgr_image(color_frame)
        print("[6] converted")
        return img


def main():
    print("=== Loading SAM3 Model ===")
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    print("=== SAM3 Loaded ===")

    cam = OrbbecCamera()
    cam.open()

    try:
        while True:
            frame = cam.grab_frame()
            if frame is None:
                continue

            # 먼저 GUI 문제 분리
            cv2.imwrite("debug_frame.jpg", frame)
            print("saved debug_frame.jpg")

            # 여기까지 되면 그 다음에만 SAM3
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            print("[7] set_image")
            state = processor.set_image(pil_image)
            print("[8] set_image done")
            print("[9] before imshow")
            # cv2.imshow("debug", frame)
            print("[10] after imshow")
            k = cv2.waitKey(1)
            print("[11] after waitKey", k)
            cv2.imshow("debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()