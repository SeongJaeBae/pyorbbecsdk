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
CONCEPTS = ["food packet"]

SEND_DEPTH = True
SEND_PRED = True
SEND_POINTCLOUD = True
SEND_ONLY_WHEN_OBJECT = False   # True면 객체 있을 때만 pred/label/depth/pcd 전송

# ---- 캡처/전송 주기 ----
CAPTURE_FPS = 5                  # 카메라 폴링 속도 상한(CPU 보호용)
CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS

RGB_SEND_FPS = 5                 # RGB만 서버로 스트리밍하는 속도
RGB_SEND_INTERVAL = 1.0 / RGB_SEND_FPS

INFER_INTERVAL_SEC = 1.0          # SAM3 추론 + pcd + pred 전송 주기
SAVE_INTERVAL_SEC = 1.0           # sam3_live_output 로컬 저장 주기

# 1초 주기 트리거 시점에 모션이 없으면 추론을 건너뛸지 여부
# False로 바꾸면 모션과 무관하게 매 INFER_INTERVAL_SEC마다 무조건 추론합니다.
REQUIRE_MOTION_FOR_INFER = True

# ---- 카메라 해상도/fps (하드웨어 프로파일 선택) ----
# 실제로 이 조합을 지원하는지는 카메라 모델마다 다르므로,
# 실행 로그의 "[CAM] Color: ..." 라인에서 실제 선택된 값을 꼭 확인할 것.
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
CAM_FPS = 10       # 예: 30. 10처럼 지원하지 않는 값이면 자동으로 기본 프로파일로 폴백됨.

# ---- 컬러 카메라 노출(Exposure) 설정 ----
# COLOR_AUTO_EXPOSURE = True  : 자동 노출(AE) 사용. 아래 수동 값들은 무시됨.
# COLOR_AUTO_EXPOSURE = False : 자동 노출을 끄고 COLOR_EXPOSURE_VALUE(+선택적으로
#                                COLOR_GAIN_VALUE)를 수동으로 적용.
# 컨베이어 벨트처럼 조명이 일정한 환경에서는 AE를 끄고 고정값을 쓰는 편이
# 프레임마다 밝기가 흔들리지 않아 SAM3/OCR 인식에 유리한 경우가 많습니다.
# COLOR_AUTO_EXPOSURE = True
# COLOR_EXPOSURE_VALUE = 300      # AE OFF일 때 사용할 수동 노출값 (raw property 값,
#                                  # 실제 밝기 단위는 장비/SDK 문서 기준. 값을 올릴수록 밝아짐)

                                 
COLOR_AUTO_EXPOSURE = False
COLOR_EXPOSURE_VALUE = 1500      # AE OFF일 때 사용할 수동 노출값 (raw property 값,
                                 # 실제 밝기 단위는 장비/SDK 문서 기준. 값을 올릴수록 밝아짐)
COLOR_GAIN_VALUE = None         # None이면 게인은 건드리지 않음. 정수를 주면 수동 게인 적용.

USE_AUTOCAST = True
AUTOCAST_DTYPE = torch.bfloat16

# ---- 모션 감지 설정 ----
MOTION_ENABLED = True
#1280
# MOTION_SENSITIVITY = 50000      # 픽셀 변화량 임계치(값이 클수록 둔감)
                                 # ROI 적용으로 픽셀 수가 줄어서 기존 56000에서 비례 축소
                                 # (56000 * 567*718 / 1280*720 ≈ 24700) — 실측 후 재튜닝 권장

                                 
MOTION_SENSITIVITY = 50000      # 픽셀 변화량 임계치(값이 클수록 둔감)
                                 # ROI 적용으로 픽셀 수가 줄어서 기존 56000에서 비례 축소
                                 # (56000 * 567*718 / 1280*720 ≈ 24700) — 실측 후 재튜닝 권장
MOTION_THRESHOLD = 25           # 프레임 diff 이진화 임계치
BLUR_KERNEL = (21, 21)
MOTION_COOLDOWN_SEC = 0.2       # 모션 트리거 최소 간격
# 1280 * 720
# MOTION_ROI = (247, 2, 567, 718) # (x, y, w, h) - 컨베이어 벨트 영역만 모션 감지 대상으로 사용
# 1920 * 1080
MOTION_ROI = (391, 2, 822, 1078) # (x, y, w, h) - 컨베이어 벨트 영역만 모션 감지 대상으로 사용

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


# ---- 모션 감지 함수 ----
def preprocess_for_motion(frame_bgr: np.ndarray, roi=None) -> np.ndarray:
    if roi is not None:
        x, y, w, h = roi
        frame_bgr = frame_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL, 0)


def detect_motion(prev_gray: np.ndarray, curr_gray: np.ndarray):
    delta = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    score = cv2.countNonZero(thresh)
    return score, thresh


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
# 스레드 간 공유 버퍼 (motion_obb.py의 LatestFrameBuffer와 동일 패턴)
# =========================
class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.version = 0
        self.frame_data = None

    def update(self, data: dict):
        with self.lock:
            self.version += 1
            data["version"] = self.version
            self.frame_data = data

    def get(self):
        with self.lock:
            if self.frame_data is None:
                return None
            return dict(self.frame_data)


# =========================
# state.json 공유 상태 (락으로 보호)
# =========================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {}

    def update(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)
            self.state["updated_at"] = time.time()
            atomic_write_json(STATE_PATH, self.state)


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
# Orbbec 프로파일 검색 helper
# =========================
def find_video_profile(profile_list, width, height, fps, preferred_formats=None):
    """
    width/height/fps에 맞는 프로파일을 찾아서 반환. 없으면 None.

    pyorbbecsdk의 get_video_stream_profile()은 인덱스 기반 순회가 아니라
    (width, height, format, fps) 4개 인자를 모두 받아서 정확히 일치하는
    프로파일을 조회하는 함수다. 포맷을 모르면 여러 후보 포맷을 순서대로
    시도해서 맞는 걸 찾는다.
    """
    if profile_list is None:
        return None

    formats_to_try = list(preferred_formats) if preferred_formats else []

    # 이 센서의 기본 프로파일이 쓰는 포맷을 최우선 후보로 시도
    try:
        default_profile = profile_list.get_default_video_stream_profile()
        if default_profile is not None:
            default_format = default_profile.get_format()
            if default_format not in formats_to_try:
                formats_to_try.insert(0, default_format)
    except Exception:
        pass

    # 흔히 쓰이는 포맷들도 후보로 추가 (이미 있으면 중복 스킵)
    for fmt_name in ["MJPG", "RGB", "YUYV", "NV12", "Y16", "Y8", "UYVY"]:
        fmt = getattr(OBFormat, fmt_name, None)
        if fmt is not None and fmt not in formats_to_try:
            formats_to_try.append(fmt)

    for fmt in formats_to_try:
        try:
            profile = profile_list.get_video_stream_profile(width, height, fmt, fps)
            if profile is not None:
                return profile
        except OBError:
            continue
        except Exception:
            continue

    return None


def probe_supported_fps(profile_list, width, height, fps_candidates=(5, 10, 15, 24, 30, 60)):
    """
    해당 해상도에서 실제로 조회에 성공하는 fps 값들을 후보군에서 찾아 로그용으로 반환.
    (진짜 "전체 목록"은 아니고, 흔한 fps 후보 + 흔한 포맷을 시도해보는 프로브 방식)
    """
    supported = []
    for fps in fps_candidates:
        profile = find_video_profile(profile_list, width, height, fps)
        if profile is not None:
            supported.append((fps, profile.get_format()))
    return supported


# =========================
# Orbbec Camera Wrapper
# =========================
class OrbbecCamera:
    def __init__(self, warmup_frames=15, timeout_ms=1000):
        self.pipeline = None
        self.config = None
        self.device = None
        self.started = False
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self.has_color = False
        self.has_depth = False

    def open(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.device = self.pipeline.get_device()

        try:
            # ---- Color ----
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is not None:
                probed = probe_supported_fps(profile_list, CAM_WIDTH, CAM_HEIGHT)
                log_line(f"[CAM] Color {CAM_WIDTH}x{CAM_HEIGHT} probed fps/format: {probed}")

                color_profile = find_video_profile(profile_list, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
                if color_profile is None:
                    log_line(
                        f"[WARN] Color {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}fps not supported, "
                        f"falling back to default profile"
                    )
                    color_profile = profile_list.get_default_video_stream_profile()

                self.config.enable_stream(color_profile)
                self.has_color = True
                log_line(
                    f"[CAM] Color selected: {color_profile.get_width()}x{color_profile.get_height()} "
                    f"@ {color_profile.get_fps()}fps"
                )

            # ---- Depth ----
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile = find_video_profile(depth_profile_list, CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
                if depth_profile is None:
                    log_line(
                        f"[WARN] Depth {CAM_WIDTH}x{CAM_HEIGHT}@{CAM_FPS}fps not supported, "
                        f"falling back to default profile"
                    )
                    depth_profile = depth_profile_list.get_default_video_stream_profile()

                self.config.enable_stream(depth_profile)
                self.has_depth = True
                log_line(
                    f"[CAM] Depth selected: {depth_profile.get_width()}x{depth_profile.get_height()} "
                    f"@ {depth_profile.get_fps()}fps"
                )

            self.config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )

        except OBError as e:
            raise RuntimeError(f"Orbbec stream profile/config 실패: {e}")

        self.pipeline.start(self.config)
        self.started = True

        # 스트림이 시작되어 device가 활성화된 상태에서 노출을 설정한다.
        # (워밍업 프레임을 버리기 전에 적용해야 워밍업 동안 노출이 안정된다)
        self._apply_exposure_settings()

        for _ in range(self.warmup_frames):
            try:
                self.pipeline.wait_for_frames(self.timeout_ms)
            except Exception:
                pass

        if not self.has_color:
            raise RuntimeError("Orbbec COLOR_SENSOR가 없습니다.")

        log_line("Orbbec camera opened.")

    def _apply_exposure_settings(self):
        """
        Config의 COLOR_AUTO_EXPOSURE / COLOR_EXPOSURE_VALUE / COLOR_GAIN_VALUE에
        따라 컬러 카메라 노출을 설정한다. 디바이스 프로퍼티 호출이라 프레임
        단위 호출(get_format/as_video_frame 등)과는 무관하게 안전하지만,
        모델/펌웨어에 따라 프로퍼티 자체가 지원되지 않을 수 있으므로 실패해도
        파이프라인 전체가 죽지 않도록 각 호출을 개별적으로 try/except 한다.
        """
        if self.device is None:
            log_line("[CAM][WARN] device가 없어 노출 설정을 건너뜁니다.")
            return

        try:
            self.device.set_bool_property(
                OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                bool(COLOR_AUTO_EXPOSURE),
            )
            log_line(f"[CAM] Auto exposure = {COLOR_AUTO_EXPOSURE}")
        except Exception as e:
            log_line(f"[CAM][WARN] Auto exposure 설정 실패: {e}")

        if not COLOR_AUTO_EXPOSURE:
            if COLOR_EXPOSURE_VALUE is not None:
                try:
                    self.device.set_int_property(
                        OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
                        int(COLOR_EXPOSURE_VALUE),
                    )
                    log_line(f"[CAM] Manual exposure = {COLOR_EXPOSURE_VALUE}")
                except Exception as e:
                    log_line(f"[CAM][WARN] 수동 exposure 설정 실패: {e}")

            if COLOR_GAIN_VALUE is not None:
                try:
                    self.device.set_int_property(
                        OBPropertyID.OB_PROP_COLOR_GAIN_INT,
                        int(COLOR_GAIN_VALUE),
                    )
                    log_line(f"[CAM] Gain = {COLOR_GAIN_VALUE}")
                except Exception as e:
                    log_line(f"[CAM][WARN] Gain 설정 실패: {e}")

    def get_exposure_status(self):
        """
        현재 디바이스에서 실제로 적용 중인 노출 관련 값을 읽어서 dict로 반환한다.
        (설정값이 아니라 device.get_*_property()로 조회한 '실측' 값)
        AE가 켜져 있으면 exposure/gain도 카메라가 자동으로 계속 바꾸므로,
        state.json을 볼 때 실시간 값을 확인할 수 있도록 매번 새로 조회한다.
        읽기가 실패하는 항목은 None으로 채워서 반환한다 (파이프라인은 안 죽음).
        """
        status = {
            "auto_exposure": None,
            "exposure": None,
            "gain": None,
            "configured_auto_exposure": bool(COLOR_AUTO_EXPOSURE),
            "configured_exposure_value": COLOR_EXPOSURE_VALUE,
            "configured_gain_value": COLOR_GAIN_VALUE,
        }

        if self.device is None:
            return status

        try:
            status["auto_exposure"] = bool(
                self.device.get_bool_property(
                    OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
                )
            )
        except Exception:
            pass

        try:
            status["exposure"] = int(
                self.device.get_int_property(
                    OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
                )
            )
        except Exception:
            pass

        try:
            status["gain"] = int(
                self.device.get_int_property(
                    OBPropertyID.OB_PROP_COLOR_GAIN_INT
                )
            )
        except Exception:
            pass

        return status

    def close(self):
        if self.pipeline and self.started:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self.started = False
        self.pipeline = None
        self.config = None
        self.device = None
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
# Capture Thread: 프레임 획득, 모션 감지, RGB 10fps 전송, 로컬 저장, 1초 주기 추론 패키지 적재
# =========================
def capture_loop(
    cam: OrbbecCamera,
    sender: PersistentSender,
    point_cloud_filter,
    frame_buffer: LatestFrameBuffer,
    shared_state: SharedState,
    stop_event: threading.Event,
):
    prev_gray = None
    last_motion_time = 0.0

    last_capture_time = 0.0
    last_rgb_send_time = 0.0
    last_save_time = 0.0
    last_infer_trigger_time = 0.0

    while not stop_event.is_set():
        now = time.time()
        if now - last_capture_time < CAPTURE_INTERVAL:
            time.sleep(0.001)
            continue
        last_capture_time = now

        try:
            frame, depth_frame, frames = cam.grab_frames()
        except Exception as e:
            log_line(f"[CAPTURE ERROR] {e}")
            time.sleep(0.01)
            continue

        if frame is None:
            continue

        # ---- 모션 감지 (MOTION_ROI 영역만 대상으로) ----
        motion_detected = False
        motion_score = 0

        if MOTION_ENABLED:
            curr_gray = preprocess_for_motion(frame, roi=MOTION_ROI)
            if prev_gray is None:
                prev_gray = curr_gray
            else:
                motion_score, _ = detect_motion(prev_gray, curr_gray)
                prev_gray = curr_gray

                if motion_score > MOTION_SENSITIVITY and \
                   (now - last_motion_time) >= MOTION_COOLDOWN_SEC:
                    motion_detected = True
                    last_motion_time = now
        else:
            motion_detected = True

        depth_vis = None
        if SEND_DEPTH and depth_frame is not None:
            depth_vis = depth_frame_to_colormap(depth_frame)

        # ---- 로컬 sam3_live_output 저장 (항상, SAVE_INTERVAL_SEC 주기) ----
        if now - last_save_time >= SAVE_INTERVAL_SEC:
            atomic_write_image(COLOR_PATH, frame)
            if depth_vis is not None:
                atomic_write_image(DEPTH_PATH, depth_vis)

            exposure_status = cam.get_exposure_status()

            shared_state.update(
                motion_enabled=MOTION_ENABLED,
                motion_score=motion_score,
                motion_detected=motion_detected,
                exposure=exposure_status,
            )
            last_save_time = now

        # ---- RGB만 서버로 스트리밍 (모션 있을 때만, 최대 10fps) ----
        if motion_detected and (now - last_rgb_send_time >= RGB_SEND_INTERVAL):
            ok, encoded_color = cv2.imencode(".jpg", frame)
            if ok:
                capture_id = datetime.now().isoformat(timespec="milliseconds")
                sender.send(
                    encoded_color.tobytes(),
                    f"color_{capture_id}.jpg",
                    "color",
                    capture_id
                )
            last_rgb_send_time = now

        # ---- 1초마다 SAM3 추론용 패키지 적재 ----
        if now - last_infer_trigger_time >= INFER_INTERVAL_SEC:
            last_infer_trigger_time = now

            if REQUIRE_MOTION_FOR_INFER and not motion_detected:
                # 이 틱에서는 모션이 없어 추론 패키지를 만들지 않음
                continue

            pc_ply_bytes = None
            if SEND_POINTCLOUD and frames is not None:
                try:
                    pc_frame = point_cloud_filter.process(frames)
                    if pc_frame is not None:
                        pc_ply_bytes = point_cloud_to_ply_bytes(pc_frame)
                except Exception as e:
                    log_line(f"[POINTCLOUD ERROR] {e}")

            frame_buffer.update({
                "color_img": frame.copy(),
                "depth_vis": depth_vis.copy() if depth_vis is not None else None,
                "pc_ply_bytes": pc_ply_bytes,
                "motion_score": motion_score,
                "motion_detected": motion_detected,
                "ts": now,
            })


# =========================
# Infer Thread: SAM3 추론 + pred/label/depth/pcd 전송
# =========================
def infer_loop(
    processor,
    sender: PersistentSender,
    frame_buffer: LatestFrameBuffer,
    shared_state: SharedState,
    model_load_time: float,
    stop_event: threading.Event,
):
    last_processed_version = -1

    pred_fps_count = 0
    pred_fps_last = time.time()
    current_pred_fps = 0

    while not stop_event.is_set():
        item = frame_buffer.get()
        if item is None or item["version"] == last_processed_version:
            time.sleep(0.01)
            continue

        last_processed_version = item["version"]
        color_img = item["color_img"]
        depth_vis = item["depth_vis"]
        pc_ply_bytes = item["pc_ply_bytes"]
        motion_score = item["motion_score"]

        try:
            all_boxes, all_masks, all_scores, set_image_time, prompt_time = run_sam3(
                processor, color_img
            )
        except Exception as e:
            log_line(f"[INFER ERROR] {e}")
            log_line(traceback.format_exc())
            continue

        label_str = build_detection_label_text(all_boxes, all_scores)
        has_object = len(all_boxes) > 0

        now = time.time()
        pred_fps_count += 1
        if now - pred_fps_last >= 1.0:
            current_pred_fps = pred_fps_count
            log_line(
                f"[PRED FPS] {pred_fps_count} | "
                f"motion={motion_score} | "
                f"set_image={set_image_time*1000:.1f}ms | "
                f"prompt={prompt_time*1000:.1f}ms | "
                f"objects={len(all_boxes)}"
            )
            pred_fps_count = 0
            pred_fps_last = now

        pred_vis = overlay_sam3_results(
            color_img,
            all_masks,
            all_boxes,
            all_scores,
            alpha=0.35,
            score_thresh=SCORE_THRESH,
            draw_boxes=True,
            show_scores=True
        )

        # ---- 로컬 저장: pred/label ----
        atomic_write_image(PRED_PATH, pred_vis)
        atomic_write_text(LABEL_PATH, label_str)

        shared_state.update(
            status="running",
            model_load_time_ms=round(model_load_time * 1000, 1),
            pred_fps=current_pred_fps,
            set_image_ms=round(set_image_time * 1000, 1),
            prompt_ms=round(prompt_time * 1000, 1),
            objects=len(all_boxes),
            concepts=CONCEPTS,
            has_object=has_object,
        )

        # ---- 서버 전송: pred/label/depth/pointcloud ----
        if (not SEND_ONLY_WHEN_OBJECT) or has_object:
            capture_id = datetime.now().isoformat(timespec="milliseconds")

            if SEND_PRED:
                ok, encoded_pred = cv2.imencode(".jpg", pred_vis)
                if ok:
                    sender.send(
                        encoded_pred.tobytes(),
                        f"pred_{capture_id}.jpg",
                        "prediction",
                        capture_id
                    )

            if label_str.strip():
                sender.send(
                    label_str.encode("utf-8"),
                    f"label_{capture_id}.txt",
                    "label",
                    capture_id
                )

            if SEND_DEPTH and depth_vis is not None:
                ok, encoded_depth = cv2.imencode(".jpg", depth_vis)
                if ok:
                    sender.send(
                        encoded_depth.tobytes(),
                        f"depth_{capture_id}.jpg",
                        "depth",
                        capture_id
                    )

            if SEND_POINTCLOUD and pc_ply_bytes is not None:
                sender.send(
                    pc_ply_bytes,
                    f"cloud_{capture_id}.ply",
                    "pointcloud",
                    capture_id
                )


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

    frame_buffer = LatestFrameBuffer()
    shared_state = SharedState()
    stop_event = threading.Event()

    shared_state.update(
        status="idle",
        model_load_time_ms=round(model_load_time * 1000, 1),
        pred_fps=0,
        set_image_ms=0,
        prompt_ms=0,
        objects=0,
        concepts=CONCEPTS,
        motion_enabled=MOTION_ENABLED,
        motion_score=0,
        motion_detected=False,
        exposure=cam.get_exposure_status(),
    )

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(cam, sender, point_cloud_filter, frame_buffer, shared_state, stop_event),
        daemon=True,
    )
    infer_thread = threading.Thread(
        target=infer_loop,
        args=(processor, sender, frame_buffer, shared_state, model_load_time, stop_event),
        daemon=True,
    )

    capture_thread.start()
    infer_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log_line("KeyboardInterrupt received. Exiting...")
    except Exception as e:
        log_line(f"[FATAL] {e}")
        log_line(traceback.format_exc())
        shared_state.update(status="error", error=str(e))
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        infer_thread.join(timeout=2.0)
        sender.close()
        cam.close()


if __name__ == "__main__":
    main()
