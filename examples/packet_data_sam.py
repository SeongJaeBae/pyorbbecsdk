import time
import cv2
import numpy as np
from PIL import Image
import torch
import socket
import struct
import threading
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

SCORE_THRESH = 0.10
MASK_THRESH = 0.5
MASK_ALPHA = 0.35
CONCEPTS = ["packet"]

SEND_DEPTH = True
SEND_ONLY_WHEN_OBJECT = False   # True면 객체 있을 때만 전송
SEND_INTERVAL_SEC = 1.0         # 1초에 1번만 전송

WINDOW_NAME = "Orbbec + SAM3"


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
            # Color stream
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                color_profile = profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.has_color = True

            # Depth stream
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile = depth_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(depth_profile)
                self.has_depth = True

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
# Utils
# =========================
def log_line(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


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

            # mask size 기준으로 box 스케일 보정
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


# =========================
# Main
# =========================
def main():
    print("=== Loading SAM3 Model ===")
    load_start = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    load_time = time.time() - load_start
    print(f"=== SAM3 Loaded: {load_time:.3f}s ===")

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    sender = PersistentSender(SERVER_IP, SERVER_PORT, log_callback=log_line)

    cam.open()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last_send_time = 0.0

    try:
        while True:
            loop_start = time.time()

            frame, depth_frame, frames = cam.grab_frames()
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # set_image
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inference_state = processor.set_image(pil_image)
            set_image_time = time.time() - t0

            all_boxes = []
            all_masks = []
            all_scores = []

            # text prompt inference
            t1 = time.time()
            for concept in CONCEPTS:
                with torch.autocast("cuda", dtype=torch.bfloat16):
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

            # visualization
            vis = overlay_sam3_results(
                frame,
                all_masks,
                all_boxes,
                all_scores,
                alpha=MASK_ALPHA,
                score_thresh=SCORE_THRESH,
                draw_boxes=True,
                show_scores=True
            )

            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0.0

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
                f"set_image: {set_image_time*1000:.1f} ms",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                vis,
                f"prompt: {prompt_time*1000:.1f} ms",
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

            cv2.imshow(WINDOW_NAME, vis)

            # =========================
            # 1초에 1번 서버 전송
            # =========================
            now = time.time()
            should_send = (now - last_send_time) >= SEND_INTERVAL_SEC
            has_object = len(all_boxes) > 0

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
                ok, encoded_pred = cv2.imencode(".jpg", vis)
                if ok:
                    sender.send(
                        encoded_pred.tobytes(),
                        f"pred_{capture_id}.jpg",
                        "prediction",
                        capture_id
                    )

                # 3) label 전송
                label_str = build_detection_label_text(all_boxes, all_scores)
                if label_str.strip():
                    sender.send(
                        label_str.encode("utf-8"),
                        f"label_{capture_id}.txt",
                        "label",
                        capture_id
                    )

                # 4) depth 전송
                if SEND_DEPTH and depth_frame is not None:
                    depth_vis = depth_frame_to_colormap(depth_frame)
                    if depth_vis is not None:
                        ok, encoded_depth = cv2.imencode(".jpg", depth_vis)
                        if ok:
                            sender.send(
                                encoded_depth.tobytes(),
                                f"depth_{capture_id}.jpg",
                                "depth",
                                capture_id
                            )

                last_send_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        sender.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()