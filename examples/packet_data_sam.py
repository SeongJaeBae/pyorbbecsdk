import os
import cv2
import time
import json
import numpy as np
from PIL import Image
import torch
import socket
import struct
import threading
import traceback
from datetime import datetime

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from pyorbbecsdk import *
from utils import frame_to_bgr_image


# =========================
# Config
# =========================
SERVER_IP = "192.168.1.142"
SERVER_PORT = 9000
CAMERA_TYPE = "orbbec"

OUTPUT_DIR = "./sam3_live_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_PATH = os.path.join(OUTPUT_DIR, "latest_color.jpg")
DEPTH_PATH = os.path.join(OUTPUT_DIR, "latest_depth.jpg")
PRED_PATH = os.path.join(OUTPUT_DIR, "latest_pred.jpg")
LABEL_PATH = os.path.join(OUTPUT_DIR, "latest_label.txt")
STATE_PATH = os.path.join(OUTPUT_DIR, "state.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "viewer.log")

SCORE_THRESH = 0.10
MASK_THRESH = 0.5
CONCEPTS = ["mealkit packet"]

SEND_DEPTH = True
SEND_PRED = True
SEND_POINTCLOUD = True
SEND_ONLY_WHEN_OBJECT = False   # True면 객체 있을 때만 서버 전송
SEND_INTERVAL_SEC = 1.0         # 서버 전송 주기
SAVE_INTERVAL_SEC = 1.0         # output dir 저장 주기

USE_AUTOCAST = True
AUTOCAST_DTYPE = torch.bfloat16


# =========================
# Utils
# =========================
def log_line(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
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
    if torch.is_tensor(x):
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return np.array(x)


def depth_frame_to_colormap(depth_frame):
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


def build_detection_label_text(boxes, scores):
    """
    형식:
    class_id x1 y1 x2 y2 score
    """
    lines = []

    for i, box in enumerate(boxes):
        score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        x1, y1, x2, y2 = map(float, box[:4])
        line = f"0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}"
        lines.append(line)

    if not lines:
        return ""
    return "\n".join(lines) + "\n"


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

        orig_h, orig_w = mask_np.shape[:2]
        mask_bin = (mask_np > MASK_THRESH).astype(np.uint8)

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

            if orig_w > 0 and orig_h > 0:
                sx = w / float(orig_w)
                sy = h / float(orig_h)
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


def point_cloud_to_ply_bytes(pc_frame):
    if pc_frame is None:
        return None

    data = pc_frame.get_data()
    if data is None:
        return None

    points_np = np.frombuffer(data, dtype=np.float32)
    if points_np.size == 0:
        return None

    # RGB_POINT -> x y z r g b(float32 6개) 형태 가정
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


# =========================
# TCP Sender
# =========================
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
                header_bytes = header.encode("utf-8")

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
        self.has_depth = False

    def open(self):
        self.pipeline = Pipeline()
        self.config = Config()

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.has_color = True

            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile = depth_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(depth_profile)
                self.has_depth = True

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

        log_line("Orbbec camera opened.")

    def close(self):
        if self.pipeline and self.started:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.started = False
        self.pipeline = None
        self.config = None
        log_line("Orbbec camera closed.")

    def grab_frames(self):
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            return None, None, None

        img = frame_to_bgr_image(color_frame)
        return img, depth_frame, frames


# =========================
# SAM3 Inference
# =========================
def run_sam3(processor, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    t0 = time.time()
    if torch.cuda.is_available() and USE_AUTOCAST:
        with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
            inference_state = processor.set_image(pil_image)
    else:
        inference_state = processor.set_image(pil_image)
    set_image_time = time.time() - t0

    all_boxes = []
    all_masks = []
    all_scores = []

    t1 = time.time()
    for concept in CONCEPTS:
        if torch.cuda.is_available() and USE_AUTOCAST:
            with torch.autocast("cuda", dtype=AUTOCAST_DTYPE):
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )
        else:
            output = processor.set_text_prompt(
                state=inference_state,
                prompt=concept
            )

        boxes = output.get("boxes", [])
        masks = output.get("masks", [])
        scores = output.get("scores", [])

        if torch.is_tensor(boxes):
            boxes = boxes.detach().float().cpu().numpy()
        else:
            boxes = to_numpy(boxes)

        if torch.is_tensor(scores):
            scores = scores.detach().float().cpu().numpy()
        else:
            scores = to_numpy(scores)

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

    return all_boxes, all_masks, all_scores, set_image_time, prompt_time


# =========================
# Main
# =========================
def main():
    log_line("Loading SAM3 model...")
    load_start = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    model_load_time = time.time() - load_start
    log_line(f"SAM3 loaded in {model_load_time:.3f}s")

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    sender = PersistentSender(SERVER_IP, SERVER_PORT, log_callback=log_line)

    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    cam.open()

    last_send_time = 0.0
    last_save_time = 0.0

    pred_fps_count = 0
    pred_fps_last = time.time()
    current_pred_fps = 0

    atomic_write_json(STATE_PATH, {
        "status": "running",
        "model_load_time_ms": round(model_load_time * 1000, 1),
        "pred_fps": 0,
        "set_image_ms": 0,
        "prompt_ms": 0,
        "objects": 0,
        "concepts": CONCEPTS,
        "updated_at": time.time(),
    })

    try:
        while True:
            loop_start = time.time()

            frame, depth_frame, frames = cam.grab_frames()
            if frame is None:
                continue

            all_boxes, all_masks, all_scores, set_image_time, prompt_time = run_sam3(
                processor, frame
            )

            label_str = build_detection_label_text(all_boxes, all_scores)
            has_object = len(all_boxes) > 0

            depth_vis = None
            if SEND_DEPTH and depth_frame is not None:
                depth_vis = depth_frame_to_colormap(depth_frame)

            pred_fps_count += 1
            now = time.time()
            if now - pred_fps_last >= 1.0:
                current_pred_fps = pred_fps_count
                log_line(
                    f"[PRED FPS] {pred_fps_count} | "
                    f"set_image={set_image_time*1000:.1f}ms | "
                    f"prompt={prompt_time*1000:.1f}ms | "
                    f"objects={len(all_boxes)}"
                )
                pred_fps_count = 0
                pred_fps_last = now

            pred_vis = overlay_sam3_results(
                frame,
                all_masks,
                all_boxes,
                all_scores,
                alpha=0.35,
                score_thresh=SCORE_THRESH,
                draw_boxes=True,
                show_scores=True
            )

            # =========================
            # output dir 저장
            # =========================
            should_save = (now - last_save_time) >= SAVE_INTERVAL_SEC

            if should_save:
                atomic_write_image(COLOR_PATH, frame)
                atomic_write_image(PRED_PATH, pred_vis)

                if depth_vis is not None:
                    atomic_write_image(DEPTH_PATH, depth_vis)

                atomic_write_text(LABEL_PATH, label_str)

                atomic_write_json(STATE_PATH, {
                    "status": "running",
                    "model_load_time_ms": round(model_load_time * 1000, 1),
                    "pred_fps": current_pred_fps,
                    "set_image_ms": round(set_image_time * 1000, 1),
                    "prompt_ms": round(prompt_time * 1000, 1),
                    "objects": len(all_boxes),
                    "concepts": CONCEPTS,
                    "has_object": has_object,
                    "updated_at": time.time(),
                })

                last_save_time = now

            # =========================
            # 서버 전송
            # =========================
            should_send = (now - last_send_time) >= SEND_INTERVAL_SEC
            if should_send and ((not SEND_ONLY_WHEN_OBJECT) or has_object):
                capture_id = datetime.now().isoformat(timespec="milliseconds")

                # 1) color 전송
                ok, encoded_color = cv2.imencode(".jpg", frame)
                if ok:
                    sender.send(
                        encoded_color.tobytes(),
                        f"color_{capture_id}.jpg",
                        "color",
                        capture_id
                    )

                # 2) prediction 전송
                if SEND_PRED:
                    ok, encoded_pred = cv2.imencode(".jpg", pred_vis)
                    if ok:
                        sender.send(
                            encoded_pred.tobytes(),
                            f"pred_{capture_id}.jpg",
                            "prediction",
                            capture_id
                        )

                # 3) label 전송
                if label_str.strip():
                    sender.send(
                        label_str.encode("utf-8"),
                        f"label_{capture_id}.txt",
                        "label",
                        capture_id
                    )

                # 4) depth 전송
                if SEND_DEPTH and depth_vis is not None:
                    ok, encoded_depth = cv2.imencode(".jpg", depth_vis)
                    if ok:
                        sender.send(
                            encoded_depth.tobytes(),
                            f"depth_{capture_id}.jpg",
                            "depth",
                            capture_id
                        )

                # 5) point cloud(.ply) 전송
                if SEND_POINTCLOUD and frames is not None:
                    try:
                        pc_frame = point_cloud_filter.process(frames)
                        if pc_frame is not None:
                            ply_bytes = point_cloud_to_ply_bytes(pc_frame)
                            if ply_bytes is not None:
                                sender.send(
                                    ply_bytes,
                                    f"cloud_{capture_id}.ply",
                                    "pointcloud",
                                    capture_id
                                )
                    except Exception as e:
                        log_line(f"[POINTCLOUD ERROR] {e}")

                last_send_time = now

            _ = time.time() - loop_start

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
        sender.close()
        cam.close()


if __name__ == "__main__":
    main()