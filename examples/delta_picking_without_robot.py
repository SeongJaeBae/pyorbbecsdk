import struct
import socket  # (남겨둬도 무방. 로봇 부분 제거했지만 혹시 다른 곳에서 쓸까봐)
import time
import cv2
import os
import json
import math
import numpy as np
import signal
from ultralytics import YOLO

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

            # Depth stream (사용 안해도 동기 안정성 위해 켜둘 수 있음)
            depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list is not None:
                depth_profile: VideoStreamProfile = depth_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(depth_profile)

            # color/depth 동기 프레임 모드
            self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

        except OBError as e:
            raise RuntimeError(f"Orbbec stream profile/config 실패: {e}")

        self.pipeline.start(self.config)
        self.started = True

        # warm-up
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
# Main (로봇 연결/명령 제거, 결과만 확인)
# =========================
def main():
    model_path = "/home/nvidia/workspace/pyorbbecsdk/examples/obb_best_260109.pt"
    MATCH_DIST_THRESH = 80

    # ✅ 선형 매핑 파라미터 (탑뷰 가정)
    # target_y = A * cy + B
    # - A: pixel -> robot_y 스케일 (초기값)
    # - B: 오프셋
    A = -1.0
    B = 0.0

    model = YOLO(model_path)

    cam = OrbbecCamera(warmup_frames=15, timeout_ms=1000)
    cam.open()

    cv2.namedWindow("Delta Robot Vision System (SIM)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Delta Robot Vision System (SIM)", 900, 650)

    pick_count = 0
    next_object_id = 0
    object_states = {}
    stop_flag = {"stop": False}

    # 디버그용 최근 타겟 값 유지
    last_target = {"id": None, "cy": None, "y": None}

    def _sigint_handler(sig, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # FPS 계산
    t0 = time.time()
    fps = 0.0
    frame_cnt = 0

    try:
        while not stop_flag["stop"]:
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

            # YOLO OBB inference
            results = model(frame, conf=0.3, iou=0.5, half=True, verbose=False)
            obb = results[0].obb

            current_objects = {}
            used_prev_ids = set()
            target_y = None
            target_id = None
            target_cy = None

            if obb is not None and len(obb) > 0:
                xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
                centers = xyxyxyxy.mean(axis=1)

                for i, pts in enumerate(xyxyxyxy):
                    cx, cy = centers[i]
                    center = (cx, cy)

                    # 이전 프레임 객체와 매칭
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

                        # ✅ 중앙선 crossing 시점에 target_y 산출 (BEV 없음)
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

                        state["center"] = center
                        state["last_side"] = current_side
                        current_objects[best_id] = state
                        used_prev_ids.add(best_id)
                        display_id = best_id

                    # Draw OBB + ID
                    pts_int = pts.astype(int)
                    cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{display_id}", (int(cx), int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Draw center y (debug)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 0), -1)
                    cv2.putText(frame, f"cy:{cy:.0f}", (int(cx) + 10, int(cy) + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            object_states = current_objects

            # UI overlay
            cv2.line(frame, (int(mid_x), 0), (int(mid_x), h), (0, 255, 255), 2)
            cv2.putText(frame, f"SIM Picks: {pick_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Mapping: target_y = {A:.3f} * cy + {B:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # crossing 발생 시, 화면에 타겟 표시 + 카운트 증가(로봇은 안 움직임)
            if target_y is not None:
                pick_count += 1
                last_target["id"] = target_id
                last_target["cy"] = target_cy
                last_target["y"] = float(target_y)

            if last_target["y"] is not None:
                cv2.putText(frame, f"LAST TARGET -> ID:{last_target['id']}  cy:{last_target['cy']:.1f}  y:{last_target['y']:.1f}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            cv2.imshow("Delta Robot Vision System (SIM)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c'):
                object_states.clear()
                print("[SIM] object_states cleared")
            elif key == ord('s'):
                # 캘리브레이션용: 마지막 타겟 값 저장/출력
                if last_target["y"] is not None:
                    print(f"[SAVE] ID={last_target['id']} cy={last_target['cy']:.2f} -> target_y={last_target['y']:.2f}")

    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"총 (SIM) 피킹 트리거: {pick_count}회")


if __name__ == "__main__":
    main()