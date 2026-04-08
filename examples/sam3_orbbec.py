import time
import cv2
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from pyorbbecsdk import *
from utils import frame_to_bgr_image


# =========================
# Orbbec Camera Wrapper
# =========================
class OrbbecCamera:
    def __init__(self, warmup_frames=15, timeout_ms=1000):
        self.pipeline = None
        self.config = None
        self.started = False
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self.has_color = False

    def open(self):
        self.pipeline = Pipeline()
        self.config = Config()

        try:
            # Color stream
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.has_color = True

            # Depth stream (필수는 아니지만 원본 구조 유지)
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile: VideoStreamProfile = depth_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(depth_profile)

            # color/depth 동기 프레임
            self.config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )

        except OBError as e:
            raise RuntimeError(f"Orbbec stream profile/config 실패: {e}")

        self.pipeline.start(self.config)
        self.started = True

        for _ in range(self.warmup_frames):
            try:
                self.pipeline.wait_for_frames(self.timeout_ms)
            except Exception:
                pass

        if not self.has_color:
            raise RuntimeError("Orbbec COLOR_SENSOR가 없습니다.")

        print("Orbbec camera opened.")

    def close(self):
        if self.pipeline and self.started:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.started = False
        self.pipeline = None
        self.config = None
        print("Orbbec camera closed.")

    def grab_frame(self):
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        img = frame_to_bgr_image(color_frame)
        return img


# =========================
# Utils
# =========================
def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def overlay_sam3_results(frame_bgr, masks, boxes, scores, alpha=0.35, score_thresh=0.3):
    vis = frame_bgr.copy()

    if masks is None or len(masks) == 0:
        return vis

    for i, mask in enumerate(masks):
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if score < score_thresh:
            continue

        mask_np = to_numpy(mask)
        if mask_np is None:
            continue

        mask_np = np.squeeze(mask_np)
        if mask_np.ndim != 2:
            continue

        mask_bin = (mask_np > 0.5).astype(np.uint8)

        color_mask = np.zeros_like(vis, dtype=np.uint8)
        color_mask[:, :, 1] = mask_bin * 255
        vis = cv2.addWeighted(vis, 1.0, color_mask, alpha, 0)

        contours, _ = cv2.findContours(
            (mask_bin * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = map(int, boxes[i][:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return vis


# =========================
# Main
# =========================
def main():
    SCORE_THRESH = 0.1
    concepts = ["packaged food on green conveyer"]   # 필요하면 ["food", "packaged food", "plastic bag"] 로 변경

    print("=== Loading SAM3 Model ===")
    load_start = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    load_time = time.time() - load_start
    print(f"=== SAM3 Loaded: {load_time:.3f}s ===")

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    cam.open()

    cv2.namedWindow("Orbbec + SAM3", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Orbbec + SAM3", 1280, 720)

    prev_time = time.time()

    try:
        while True:
            loop_start = time.time()

            frame = cam.grab_frame()
            if frame is None:
                continue

            # BGR -> RGB -> PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # set_image
            t0 = time.time()
            inference_state = processor.set_image(pil_image)
            set_image_time = time.time() - t0

            all_boxes = []
            all_masks = []
            all_scores = []

            # text prompt inference
            t1 = time.time()
            for concept in concepts:
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )

                boxes = to_numpy(output.get("boxes", []))
                masks = output.get("masks", [])
                scores = to_numpy(output.get("scores", []))

                if boxes is None or len(boxes) == 0:
                    continue

                if scores is None or len(scores) == 0:
                    scores = np.ones((len(boxes),), dtype=np.float32)

                for i in range(len(boxes)):
                    if float(scores[i]) >= SCORE_THRESH:
                        all_boxes.append(boxes[i])
                        all_masks.append(masks[i])
                        all_scores.append(float(scores[i]))

            prompt_time = time.time() - t1

            # visualization
            vis = overlay_sam3_results(
                frame,
                all_masks,
                all_boxes,
                all_scores,
                alpha=0.35,
                score_thresh=SCORE_THRESH
            )

            # FPS
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0.0

            # Display text
            cv2.putText(
                vis,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"set_image: {set_image_time:.3f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"prompt: {prompt_time:.3f}s",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"objects: {len(all_boxes)}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("Orbbec + SAM3", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()