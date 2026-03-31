import socket
import struct
import time
import cv2
import os
import json
import math
import numpy as np
import signal
import torch

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from pyorbbecsdk import *
from utils import frame_to_bgr_image


# =========================
# Robot Controller
# =========================
class RobotController:
    def __init__(self, ip='192.168.27.16', port=502):
        self.ip = ip
        self.port = port
        self.socket = None

        self.STATUS_QUERY_CMD = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 2D 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        self.MOVE_COMPLETE_RESPONSE = bytes.fromhex(
            '30 30 00 00 00 08 01 03 00 2D 00 00 00 01'.replace(' ', '')
        )

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5.0)
        self.socket.connect((self.ip, self.port))
        print(f"로봇 연결됨: {self.ip}:{self.port}")

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def coordinate_to_bytes(self, value):
        return struct.pack('<i', int(value * 1000))

    def create_move_command(self, x, y, z, r, is_linear=False, speed=1000, h_value=50):
        if is_linear:
            header = bytes([0x30, 0x30, 0x00, 0x00, 0x00, 0x22, 0x01, 0x06, 0x00, 0x17])
        else:
            header = bytes([0x30, 0x30, 0x00, 0x00, 0x00, 0x22, 0x01, 0x06, 0x00, 0x03])

        x_bytes = self.coordinate_to_bytes(x)
        y_bytes = self.coordinate_to_bytes(y)
        z_bytes = self.coordinate_to_bytes(z)
        r_bytes = self.coordinate_to_bytes(r)

        speed_bytes = struct.pack('<i', speed)

        if is_linear:
            h_bytes = struct.pack('<i', h_value)
            tail = speed_bytes + h_bytes
        else:
            tail = speed_bytes + bytes([0x00, 0x00, 0x00, 0x00])

        command = header + x_bytes + y_bytes + z_bytes + r_bytes + tail
        return command

    def send_command(self, byte_data, print_log=False):
        if not self.socket:
            return None
        self.socket.sendall(byte_data)
        try:
            self.socket.settimeout(2.0)
            return self.socket.recv(1024)
        except socket.timeout:
            return None

    def wait_until_complete(self, timeout=30, poll_interval=0.01):
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.send_command(self.STATUS_QUERY_CMD)
            if response == self.MOVE_COMPLETE_RESPONSE:
                return True
            time.sleep(poll_interval)
        return False

    def move(self, x, y, z, r, is_linear, speed=1000):
        cmd = self.create_move_command(x, y, z, r, is_linear, speed)
        return self.send_command(cmd)

    def home(self):
        cmd = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        return self.send_command(cmd)

    def suction_on(self):
        cmd = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 0C E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        cmd_2 = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 0C 00 00 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        self.send_command(cmd)
        return self.send_command(cmd_2)

    def suction_off(self):
        cmd1 = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 0C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        cmd2 = bytes.fromhex(
            '30 30 00 00 00 22 01 06 00 0C E8 03 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', '')
        )
        self.send_command(cmd1)
        return self.send_command(cmd2)


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
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        img = frame_to_bgr_image(color_frame)
        return img


# =========================
# Visualization Utils
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
    MATCH_DIST_THRESH = 80

    BELT_WIDTH_CM = 25.0
    PX_PER_CM = 7.6

    CAMERA_HEIGHT_MM = 500.0
    CAMERA_TO_PICK_DISTANCE_MM = 1200.0
    BELT_SPEED_MM_S = 20.0

    ROBOT_BASE_Y_MM = 520.0
    ROBOT_LEAD_TIME_S = 0.0

    SCORE_THRESH = 0.30
    concepts = ["food"]

    print("=== Loading SAM3 Model ===")
    load_start = time.time()
    sam3_model = build_sam3_image_model()
    processor = Sam3Processor(sam3_model)
    load_time = time.time() - load_start
    print(f"=== SAM3 Loaded: {load_time:.3f}s ===")

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    cam.open()

    robot = RobotController(ip='192.168.27.16', port=502)
    try:
        robot.connect()
    except Exception as e:
        print(f"로봇 연결 실패: {e}")

    cv2.namedWindow("Delta Robot Vision System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Delta Robot Vision System", 800, 600)

    pick_count = 0
    next_object_id = 0
    object_states = {}
    scheduled_pick = None
    stop_flag = {"stop": False}

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop_flag["stop"]:
            frame = cam.grab_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            mid_x = w / 2
            mid_y = h / 2

            belt_width_px = BELT_WIDTH_CM * PX_PER_CM
            half_belt_px = belt_width_px / 2.0

            top_belt_y = int(mid_y - half_belt_px)
            bottom_belt_y = int(mid_y + half_belt_px)
            belt_center_y = (top_belt_y + bottom_belt_y) / 2.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            set_image_start = time.time()
            inference_state = processor.set_image(pil_image)
            set_image_time = time.time() - set_image_start

            all_boxes = []
            all_masks = []
            all_scores = []

            prompt_start = time.time()
            for concept in concepts:
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=concept
                )

                boxes = output.get("boxes", [])
                masks = output.get("masks", [])
                scores = output.get("scores", [])

                boxes = to_numpy(boxes)
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

            process_time = time.time() - prompt_start

            current_objects = {}
            used_prev_ids = set()

            if len(all_boxes) > 0:
                centers = []
                for box in all_boxes:
                    x1, y1, x2, y2 = box[:4]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    centers.append((cx, cy))

                for i, box in enumerate(all_boxes):
                    x1, y1, x2, y2 = box[:4]
                    cx, cy = centers[i]
                    center = (cx, cy)

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
                            "center": center,
                            "last_side": current_side,
                            "captured": False
                        }
                        display_id = obj_id
                    else:
                        state = object_states[best_id]
                        current_side = "left" if cx < mid_x else "right"

                        if current_side != state["last_side"] and not state["captured"] and scheduled_pick is None:
                            dy_px = float(cy) - belt_center_y
                            dy_cm = dy_px / PX_PER_CM
                            target_y = -1 * (ROBOT_BASE_Y_MM + dy_cm)

                            time_to_pick = CAMERA_TO_PICK_DISTANCE_MM / BELT_SPEED_MM_S
                            trigger_delay = max(0.0, time_to_pick - ROBOT_LEAD_TIME_S)

                            scheduled_pick = {
                                "target_y": target_y,
                                "pick_time": trigger_delay if trigger_delay > 0 else time_to_pick,
                                "id": best_id
                            }

                            print(
                                f"CROSSING: ID={best_id}, cy={cy:.1f}, "
                                f"dy_cm={dy_cm:.2f}cm, target_y={target_y:.1f}, "
                                f"camera_height={CAMERA_HEIGHT_MM:.1f}mm, "
                                f"pick_after={time_to_pick:.2f}s"
                            )

                            state["captured"] = True

                        state["center"] = center
                        state["last_side"] = current_side
                        current_objects[best_id] = state
                        used_prev_ids.add(best_id)
                        display_id = best_id

                        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            str(display_id),
                            (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2
                        )

                frame = overlay_sam3_results(
                    frame,
                    all_masks,
                    all_boxes,
                    all_scores,
                    alpha=0.35,
                    score_thresh=SCORE_THRESH
                )

            object_states = current_objects

            cv2.line(frame, (int(mid_x), 0), (int(mid_x), h), (0, 255, 255), 2)
            cv2.line(frame, (0, top_belt_y), (w, top_belt_y), (0, 255, 255), 2)
            cv2.line(frame, (0, bottom_belt_y), (w, bottom_belt_y), (0, 255, 255), 2)

            cv2.putText(
                frame,
                f"Pick: {pick_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"SAM3 set={set_image_time:.3f}s prompt={process_time:.3f}s",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if scheduled_pick is not None:
                remain = scheduled_pick["pick_time"]
                cv2.putText(
                    frame,
                    f"Next pick in: {max(0, remain):.2f}s",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2
                )

            cv2.imshow("Delta Robot Vision System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if scheduled_pick is not None:
                target_y = scheduled_pick["target_y"]
                time_to_speed = int(max(1, scheduled_pick["pick_time"] * 100))

                pick_count += 1
                print(f"피킹 #{pick_count}, target_y={target_y:.1f}")

                robot.home()
                robot.wait_until_complete()

                robot.suction_on()
                robot.wait_until_complete()

                robot.move(x=0, y=target_y, z=-620, r=0.5, is_linear=True, speed=time_to_speed)
                robot.wait_until_complete()

                robot.move(x=48, y=500, z=-710, r=0.5, is_linear=True, speed=300)
                robot.wait_until_complete()

                robot.suction_off()
                time.sleep(0.03)

                robot.home()
                robot.wait_until_complete()

                object_states.clear()
                scheduled_pick = None
                print("피킹 완료\n")

    finally:
        robot.disconnect()
        cam.close()
        cv2.destroyAllWindows()
        print(f"총 피킹: {pick_count}회")


if __name__ == "__main__":
    main()