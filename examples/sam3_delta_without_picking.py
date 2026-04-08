import time
import math
import signal
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
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.has_color = True

            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile: VideoStreamProfile = depth_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(depth_profile)

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
            raise RuntimeError("Orbbec COLOR_SENSOR가 없습니다 (color stream enable 실패).")

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
        if not self.started:
            return None

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


def preprocess_for_motion(frame_bgr: np.ndarray, blur_kernel=(21, 21)) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, blur_kernel, 0)


def detect_motion(prev_gray: np.ndarray, curr_gray: np.ndarray, threshold_value=25):
    delta = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(delta, threshold_value, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    score = cv2.countNonZero(thresh)
    return score, thresh


def mask_to_center(mask, min_area=300):
    """
    mask -> center(cx, cy), area
    contour/minAreaRect 대신 numpy 기반 중심 계산
    """
    mask_np = to_numpy(mask)
    if mask_np is None:
        return None, 0, None

    mask_np = np.squeeze(mask_np)
    if mask_np.ndim != 2:
        return None, 0, None

    mask_bin = (mask_np > 0.5).astype(np.uint8)
    area = int(mask_bin.sum())

    if area < min_area:
        return None, area, mask_bin

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, area, mask_bin

    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy), area, mask_bin


def mask_to_bbox(mask_bin):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def overlay_mask(frame_bgr, mask_bin, alpha=0.30):
    vis = frame_bgr.copy()
    color_mask = np.zeros_like(vis, dtype=np.uint8)
    color_mask[:, :, 1] = mask_bin * 255
    vis = cv2.addWeighted(vis, 1.0, color_mask, alpha, 0)
    return vis


# =========================
# Main
# =========================
def main():
    # ---------- tracking / mapping ----------
    MATCH_DIST_THRESH = 80
    A = -1.0
    B = 520

    # ---------- SAM3 ----------
    SCORE_THRESH = 0.5
    MIN_MASK_AREA = 500
    #concept = "packaged food on green conveyer"   # 예: ["food", "packaged food", "plastic bag"]
    concept = "packet"   # 예: ["food", "packaged food", "plastic bag"]

    # ---------- motion ----------
    MOTION_ENABLED = False
    MOTION_SENSITIVITY = 3000
    MOTION_THRESHOLD = 25
    MOTION_BLUR_KERNEL = (21, 21)
    MOTION_COOLDOWN_SEC = 0.2
    SHOW_MOTION_MASK = False

    # ---------- inference skip ----------
    SAM3_FRAME_SKIP = 3   # 3프레임마다 1번 추론

    print("=== Loading SAM3 Model ===")
    load_start = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    load_time = time.time() - load_start
    print(f"=== SAM3 Loaded: {load_time:.3f}s ===")

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    cam.open()

    cv2.namedWindow("Delta Robot Vision System (SAM3 Optimized)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Delta Robot Vision System (SAM3 Optimized)", 1280, 720)

    pick_count = 0
    next_object_id = 0
    object_states = {}
    stop_flag = {"stop": False}
    last_target = {"id": None, "cy": None, "y": None}

    prev_gray = None
    last_motion_time = 0.0

    frame_cnt = 0
    fps = 0.0
    t0 = time.time()

    # 마지막 추론 결과 유지
    last_vis_detections = []
    last_set_image_time = 0.0
    last_prompt_time = 0.0
    last_total_infer_time = 0.0
    last_motion_score = 0
    last_thresh = None

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_flag["stop"]:
            loop_start = time.time()

            frame = cam.grab_frame()
            if frame is None:
                continue

            frame_cnt += 1
            if frame_cnt % 10 == 0:
                t1 = time.time()
                dt = t1 - t0
                if dt > 1e-6:
                    fps = 10.0 / dt
                t0 = t1

            h, w = frame.shape[:2]
            mid_x = w / 2

            # =========================
            # Motion detection
            # =========================
            motion_detected = True
            motion_score = 0
            thresh = None

            if MOTION_ENABLED:
                curr_gray = preprocess_for_motion(frame, blur_kernel=MOTION_BLUR_KERNEL)

                if prev_gray is None:
                    prev_gray = curr_gray
                    continue

                motion_score, thresh = detect_motion(
                    prev_gray,
                    curr_gray,
                    threshold_value=MOTION_THRESHOLD
                )
                prev_gray = curr_gray

                if (
                    motion_score > MOTION_SENSITIVITY
                    and (time.time() - last_motion_time) >= MOTION_COOLDOWN_SEC
                ):
                    motion_detected = True
                    last_motion_time = time.time()
                else:
                    motion_detected = False

            last_motion_score = motion_score
            last_thresh = thresh

            # =========================
            # SAM3 inference 조건
            # =========================
            run_sam3 = motion_detected and (frame_cnt % SAM3_FRAME_SKIP == 0)

            if run_sam3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)

                # 1) set_image
                infer_t0 = time.time()
                t_set0 = time.time()
                inference_state = processor.set_image(pil_image)
                set_image_time = time.time() - t_set0

                # 2) multi-concept 한번만 호출
                t_prompt0 = time.time()
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )
                prompt_time = time.time() - t_prompt0

                boxes = to_numpy(output.get("boxes", []))
                masks = output.get("masks", [])
                scores = to_numpy(output.get("scores", []))

                detections = []

                if boxes is not None and len(boxes) > 0:
                    if scores is None or len(scores) == 0:
                        scores = np.ones((len(boxes),), dtype=np.float32)

                    max_n = min(len(boxes), len(masks), len(scores))

                    for i in range(max_n):
                        score = float(scores[i])
                        if score < SCORE_THRESH:
                            continue

                        center, area, mask_bin = mask_to_center(
                            masks[i],
                            min_area=MIN_MASK_AREA
                        )
                        if center is None or mask_bin is None:
                            continue

                        bbox = mask_to_bbox(mask_bin)
                        if bbox is None:
                            continue

                        detections.append({
                            "score": score,
                            "mask_bin": mask_bin,
                            "bbox": bbox,
                            "center": center,
                            "area": area
                        })

                total_infer_time = time.time() - infer_t0

                last_vis_detections = detections
                last_set_image_time = set_image_time
                last_prompt_time = prompt_time
                last_total_infer_time = total_infer_time

            # =========================
            # tracking / crossing
            # =========================
            current_objects = {}
            used_prev_ids = set()
            target_y = None
            target_id = None
            target_cy = None

            vis = frame.copy()

            for det in last_vis_detections:
                cx, cy = det["center"]
                x1, y1, x2, y2 = det["bbox"]

                best_id = None
                best_dist = float("inf")

                for obj_id, state in object_states.items():
                    if obj_id in used_prev_ids:
                        continue
                    d = math.hypot(cx - state["center"][0], cy - state["center"][1])
                    if d < best_dist and d < MATCH_DIST_THRESH:
                        best_dist = d
                        best_id = obj_id

                if best_id is None:
                    obj_id = next_object_id
                    next_object_id += 1
                    current_side = "left" if cx < mid_x else "right"
                    current_objects[obj_id] = {
                        "center": (cx, cy),
                        "last_side": current_side,
                        "captured": False
                    }
                    display_id = obj_id
                else:
                    state = object_states[best_id]
                    current_side = "left" if cx < mid_x else "right"

                    if current_side != state["last_side"] and not state["captured"]:
                        target_y = A * float(cy) + B
                        target_id = best_id
                        target_cy = float(cy)

                        print(
                            f"[CROSSING] ID={best_id}  "
                            f"cx={cx:.1f} cy={cy:.1f}  "
                            f"-> target_y={target_y:.2f}"
                        )
                        state["captured"] = True

                    state["center"] = (cx, cy)
                    state["last_side"] = current_side
                    current_objects[best_id] = state
                    used_prev_ids.add(best_id)
                    display_id = best_id

                vis = overlay_mask(vis, det["mask_bin"], alpha=0.25)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (255, 255, 0), -1)

                cv2.putText(
                    vis,
                    f"ID:{display_id} S:{det['score']:.2f}",
                    (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.putText(
                    vis,
                    f"cy:{cy:.0f}",
                    (int(cx) + 10, int(cy) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            object_states = current_objects

            if target_y is not None:
                pick_count += 1
                last_target["id"] = target_id
                last_target["cy"] = target_cy
                last_target["y"] = float(target_y)

            # =========================
            # UI
            # =========================
            cv2.line(vis, (int(mid_x), 0), (int(mid_x), h), (0, 255, 255), 2)

            cv2.putText(
                vis,
                f"SIM Picks: {pick_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2
            )
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )
            cv2.putText(
                vis,
                f"Mapping: target_y = {A:.3f} * cy + {B:.3f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"Motion score: {last_motion_score}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"set_image: {last_set_image_time:.3f}s",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"prompt: {last_prompt_time:.3f}s",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"infer total: {last_total_infer_time:.3f}s",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"objects: {len(last_vis_detections)}",
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"motion: {'ON' if motion_detected else 'OFF'} / skip:{SAM3_FRAME_SKIP}",
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255) if motion_detected else (180, 180, 180),
                2
            )

            if last_target["y"] is not None:
                cv2.putText(
                    vis,
                    f"LAST TARGET -> ID:{last_target['id']}  cy:{last_target['cy']:.1f}  y:{last_target['y']:.1f}",
                    (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            cv2.imshow("Delta Robot Vision System (SAM3 Optimized)", vis)

            if SHOW_MOTION_MASK and last_thresh is not None:
                thresh_vis = cv2.resize(last_thresh, (640, 360))
                cv2.imshow("Motion Mask", thresh_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                object_states.clear()
                print("[SIM] object_states cleared")
            elif key == ord('s'):
                if last_target["y"] is not None:
                    print(
                        f"[SAVE] ID={last_target['id']} "
                        f"cy={last_target['cy']:.2f} -> target_y={last_target['y']:.2f}"
                    )

    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"총 (SIM) 피킹 트리거: {pick_count}회")


if __name__ == "__main__":
    main()
