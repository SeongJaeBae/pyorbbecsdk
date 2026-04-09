import os
import cv2
import time
import json
import traceback
import numpy as np
from PIL import Image

from pyorbbecsdk import *
from utils import frame_to_bgr_image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================================================
# Config
# =========================================================
OUTPUT_DIR = "./sam3_live_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_PATH = os.path.join(OUTPUT_DIR, "latest_pred.jpg")
DEPTH_PATH = os.path.join(OUTPUT_DIR, "latest_depth.jpg")
STATE_PATH = os.path.join(OUTPUT_DIR, "state.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "viewer.log")

CAPTURE_FPS = 15
CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS

SCORE_THRESH = 0.10
MASK_THRESH = 0.5
MASK_ALPHA = 0.35
CONCEPTS = ["packet"]


# =========================================================
# Utils
# =========================================================
def log_line(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_image(path, image):
    tmp = path + ".tmp.jpg"
    ok = cv2.imwrite(tmp, image)
    if ok:
        os.replace(tmp, path)


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


def depth_frame_to_colormap(depth_frame: DepthFrame):
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

def overlay_sam3_results(
    frame_bgr,
    masks,
    boxes=None,
    scores=None,
    alpha=0.35,
    score_thresh=0.3,
    draw_boxes=True,
    show_scores=True,
):
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

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

        # binary mask
        mask_bin = (mask_np > MASK_THRESH).astype(np.uint8)

        # ---------------------------------
        # 핵심: 대상 이미지 크기에 맞춰 resize
        # ---------------------------------
        if mask_bin.shape[0] != h or mask_bin.shape[1] != w:
            mask_bin = cv2.resize(
                mask_bin,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        color_mask = np.zeros_like(vis, dtype=np.uint8)
        color_mask[:, :, 1] = mask_bin * 255
        vis = cv2.addWeighted(vis, 1.0, color_mask, alpha, 0)

        contours, _ = cv2.findContours(
            (mask_bin * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

        if draw_boxes and boxes is not None and i < len(boxes):
            box = np.array(boxes[i][:4], dtype=np.float32)

            x1, y1, x2, y2 = box

            # ---------------------------------
            # box도 frame 크기가 다르면 스케일 보정
            # ---------------------------------
            if mask_np.shape[1] > 0 and mask_np.shape[0] > 0:
                sx = w / float(mask_np.shape[1])
                sy = h / float(mask_np.shape[0])
                x1 *= sx
                x2 *= sx
                y1 *= sy
                y2 *= sy

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if show_scores:
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
# =========================================================
# Main
# =========================================================
def main():
    log_line("Loading SAM3 model...")
    t0 = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    model_load_time = time.time() - t0
    log_line(f"SAM3 loaded in {model_load_time:.3f}s")

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    try:
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
        )
        log_line("FULL_FRAME_REQUIRE enabled")
    except Exception:
        log_line("aggregate mode setup skipped")

    pipeline.start(config)
    log_line("Pipeline started")

    last_capture_time = 0.0
    capture_fps_count = 0
    capture_fps_last = time.time()

    pred_fps_count = 0
    pred_fps_last = time.time()

    atomic_write_json(STATE_PATH, {
        "status": "running",
        "model_load_time_ms": round(model_load_time * 1000, 1),
        "capture_fps": 0,
        "pred_fps": 0,
        "set_image_ms": 0,
        "prompt_ms": 0,
        "objects": 0,
        "updated_at": time.time(),
    })

    try:
        while True:
            now = time.time()
            if now - last_capture_time < CAPTURE_INTERVAL:
                time.sleep(0.001)
                continue
            last_capture_time = now

            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame is None or depth_frame is None:
                continue

            color_img = frame_to_bgr_image(color_frame)
            if color_img is None:
                continue

            depth_vis = depth_frame_to_colormap(depth_frame)
            if depth_vis is None:
                continue

            capture_fps_count += 1
            capture_now = time.time()
            current_capture_fps = 0
            if capture_now - capture_fps_last >= 1.0:
                current_capture_fps = capture_fps_count
                log_line(f"[CAPTURE FPS] {capture_fps_count}")
                capture_fps_count = 0
                capture_fps_last = capture_now

            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            t_set = time.time()
            inference_state = processor.set_image(pil_image)
            set_image_time = time.time() - t_set

            all_boxes = []
            all_masks = []
            all_scores = []

            t_prompt = time.time()
            for concept in CONCEPTS:
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

            prompt_time = time.time() - t_prompt

            pred_vis = overlay_sam3_results(
                color_img,
                all_masks,
                all_boxes,
                all_scores,
                alpha=MASK_ALPHA,
                score_thresh=SCORE_THRESH,
                draw_boxes=True,
                show_scores=True
            )

            depth_overlay = overlay_sam3_results(
                depth_vis,
                all_masks,
                all_boxes,
                all_scores,
                alpha=MASK_ALPHA,
                score_thresh=SCORE_THRESH,
                draw_boxes=False,
                show_scores=False
            )

            cv2.putText(
                pred_vis,
                f"set_image: {set_image_time*1000:.1f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                pred_vis,
                f"prompt: {prompt_time*1000:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                pred_vis,
                f"objects: {len(all_boxes)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                depth_overlay,
                f"set_image: {set_image_time*1000:.1f} ms",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                depth_overlay,
                f"prompt: {prompt_time*1000:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                depth_overlay,
                f"objects: {len(all_boxes)}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            atomic_write_image(PRED_PATH, pred_vis)
            atomic_write_image(DEPTH_PATH, depth_overlay)

            pred_fps_count += 1
            pred_now = time.time()
            current_pred_fps = 0
            if pred_now - pred_fps_last >= 1.0:
                current_pred_fps = pred_fps_count
                log_line(
                    f"[PRED FPS] {pred_fps_count} | "
                    f"set_image={set_image_time*1000:.1f}ms | "
                    f"prompt={prompt_time*1000:.1f}ms | "
                    f"objects={len(all_boxes)}"
                )
                pred_fps_count = 0
                pred_fps_last = pred_now

            atomic_write_json(STATE_PATH, {
                "status": "running",
                "model_load_time_ms": round(model_load_time * 1000, 1),
                "capture_fps": current_capture_fps,
                "pred_fps": current_pred_fps,
                "set_image_ms": round(set_image_time * 1000, 1),
                "prompt_ms": round(prompt_time * 1000, 1),
                "objects": len(all_boxes),
                "concepts": CONCEPTS,
                "updated_at": time.time(),
            })

    except KeyboardInterrupt:
        log_line("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        log_line(f"[FATAL] {e}")
        log_line(traceback.format_exc())
        atomic_write_json(STATE_PATH, {
            "status": "error",
            "error": str(e),
            "updated_at": time.time(),
        })
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        log_line("Pipeline stopped")


if __name__ == "__main__":
    main()