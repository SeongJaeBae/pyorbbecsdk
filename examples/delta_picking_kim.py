import socket
import struct
import time
import cv2
import os
import json
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import threading
import queue
import signal
import sys
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from ultralytics import YOLO


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
    
    def create_move_command(self, x, y, z, r, speed=1000):
        header = bytes([0x30, 0x30, 0x00, 0x00, 0x00, 0x22, 0x01, 0x06, 0x00, 0x03])
        x_bytes = self.coordinate_to_bytes(x)
        y_bytes = self.coordinate_to_bytes(y)
        z_bytes = self.coordinate_to_bytes(z)
        r_bytes = self.coordinate_to_bytes(r)
        speed_bytes = struct.pack('<i', speed)
        tail = speed_bytes + bytes([0x00, 0x00, 0x00, 0x00])
        return header + x_bytes + y_bytes + z_bytes + r_bytes + tail
    
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
    
    def move(self, x, y, z, r, speed=1000):
        cmd = self.create_move_command(x, y, z, r, speed)
        return self.send_command(cmd)
    
    def home(self):
        cmd = bytes.fromhex('30 30 00 00 00 22 01 06 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', ''))
        return self.send_command(cmd)
    
    def suction_on(self):
        cmd = bytes.fromhex('30 30 00 00 00 22 01 06 00 0C E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', ''))
        cmd_2 = bytes.fromhex('30 30 00 00 00 22 01 06 00 0C 00 00 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', ''))
        self.send_command(cmd)
        return self.send_command(cmd_2)
    
    def suction_off(self):
        cmd1 = bytes.fromhex('30 30 00 00 00 22 01 06 00 0C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', ''))
        cmd2 = bytes.fromhex('30 30 00 00 00 22 01 06 00 0C E8 03 00 00 E8 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00'.replace(' ', ''))
        self.send_command(cmd1)
        return self.send_command(cmd2)


class BEVTransformer:
    def __init__(self, calib_path):
        self.src_points = None
        self.dst_points = None
        self.M = None
        self.calibrated = False
        self.src_contour = None
        self.load_calibration(calib_path)
    
    def is_inside_roi(self, point):
        if self.src_contour is None:
            return True
        result = cv2.pointPolygonTest(self.src_contour, (float(point[0]), float(point[1])), False)
        return result >= 0
    
    def load_calibration(self, path):
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
        self.src_points = np.float32(data["src_points"])
        self.dst_points = np.float32(data["dst_points"])
        self.src_contour = self.src_points.reshape(-1, 1, 2).astype(np.float32)
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.calibrated = True
    
    def image_to_bev(self, point, check_roi=True):
        if not self.calibrated:
            return None
        if check_roi and not self.is_inside_roi(point):
            return None
        point = np.array(point, dtype=np.float32).reshape(1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.M)
        return transformed.reshape(2)


model_path = "runs_keti_iksan_obb/yolo11m_obb/weights/best.pt"
BEV_CALIB_PATH = "bev_calibration.json"
MATCH_DIST_THRESH = 80

model = YOLO(model_path)
cam = HikCamera(cam_index=1)
cam.open()
bev = BEVTransformer(BEV_CALIB_PATH)

if not bev.calibrated:
    cam.close()
    exit(1)

robot = RobotController(ip='192.168.27.16', port=502)
try:
    robot.connect()
except Exception as e:
    print(f"로봇 연결 실패: {e}")

cv2.namedWindow("Delta Robot Vision System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Delta Robot Vision System", 800, 600)

frame_idx = 0
pick_count = 0
next_object_id = 0
object_states = {}

try:
    while True:
        frame = cam.grab_frame()
        if frame is None:
            continue

        frame_idx += 1
        h, w = frame.shape[:2]
        mid_x = w / 2

        results = model(frame, conf=0.3, iou=0.5, half=True, verbose=False)
        obb = results[0].obb

        current_objects = {}
        used_prev_ids = set()
        target_y = None

        if obb is not None and len(obb) > 0:
            xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
            centers = xyxyxyxy.mean(axis=1)

            for i, pts in enumerate(xyxyxyxy):
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

                    if current_side != state["last_side"] and not state["captured"]:
                        bev_result = bev.image_to_bev([cx, cy], check_roi=True)
                        if bev_result is not None:
                            bev_cx, bev_cy = bev_result
                            target_y = -1.0 * (bev_cy + 420.0)
                            print(f"CROSSING: ID={best_id}, BEV=({bev_cx:.1f}, {bev_cy:.1f}), target_y={target_y:.1f}")
                        state["captured"] = True

                    state["center"] = center
                    state["last_side"] = current_side
                    current_objects[best_id] = state
                    used_prev_ids.add(best_id)
                    display_id = best_id

                pts_int = pts.astype(int)
                cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                cv2.putText(frame, str(display_id), (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        object_states = current_objects

        cv2.line(frame, (int(mid_x), 0), (int(mid_x), h), (0, 255, 255), 2)
        cv2.putText(frame, f"Pick: {pick_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Delta Robot Vision System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        if target_y is not None:
            pick_count += 1
            print(f"피킹 #{pick_count}")

            robot.home()
            robot.wait_until_complete()

            robot.move(x=0, y=0, z=-597, r=0, speed=2850)
            robot.wait_until_complete()

            robot.move(x=48, y=target_y, z=-627, r=0, speed=1000)
            robot.wait_until_complete()
                       
            robot.suction_on()
            robot.move(x=48, y=target_y, z=-643, r=0, speed=780)
            robot.wait_until_complete()            

            robot.move(x=48, y=target_y, z=-643, r=0, speed=1200)
            robot.wait_until_complete()

            robot.move(x=48, y=target_y, z=-607, r=0, speed=800)
            robot.wait_until_complete()

            robot.move(x=48, y=500, z=-690, r=0, speed=1000)
            robot.wait_until_complete()

            robot.move(x=48, y=500, z=-710, r=0, speed=300)
            robot.wait_until_complete()

            robot.suction_off()
            time.sleep(0.03)

            robot.home()
            robot.wait_until_complete()

            object_states.clear()
            print("피킹 완료\n")
        
        
finally:
    robot.disconnect()
    cam.close()
    cv2.destroyAllWindows()
    print(f"총 피킹: {pick_count}회")