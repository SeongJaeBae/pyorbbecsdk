########################################################################
# ZED One Motion Detection Capture
# ZED 카메라에서 움직임을 감지하면 자동으로 촬영하는 통합 코드
########################################################################

import pyzed.sl as sl
import cv2
import time
import os
import uuid
import numpy as np
import socket
import struct

# ===================== 설정 =====================
SAVE_DIR        = './motion_captured'   # 저장 폴더
SENSITIVITY     = 500                  # 움직임 민감도 (작을수록 예민, 500~3000 권장)
BLUR_KERNEL     = (21, 21)             # 노이즈 제거 블러 커널 크기
COOLDOWN_SEC    = 0.1                  # 연속 저장 방지 쿨다운 (초)
DISPLAY_WIDTH   = 1280                 # 미리보기 창 가로 해상도
DISPLAY_HEIGHT  = 720                  # 미리보기 창 세로 해상도
# ================================================

os.makedirs(SAVE_DIR, exist_ok=True)



SERVER_IP = '192.168.1.154'    #공인IP
PORT = 9000                    #포트포워딩 외부포트 (19000 -> 9000)


def send_file(camera_type, file_type, frame_send, capture_id):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"sealing_{timestamp}.jpg"
    frame_np = np.asanyarray(frame_send)
    ret, buffer = cv2.imencode('.jpg', frame_np)
    if not ret:
        print("이미지 인코딩 실패")
        return
    
    image_bytes = buffer.tobytes()
    image_size = len(image_bytes)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect((SERVER_IP, PORT))
            print("b")
            header = f"{camera_type}|{file_type}|{capture_id}|{filename}".encode()
            client.sendall(struct.pack('!I', len(header)))
            client.sendall(header)
            client.sendall(struct.pack('!Q', image_size))
            client.sendall(image_bytes)
            resp = client.recv(1024).decode()
            print("서버 응답:", resp)
    except Exception as e:
        print(f"[전송에러] {e}")
        



def preprocess(frame_bgra: np.ndarray) -> np.ndarray:
    """BGRA 프레임 → 그레이스케일 + 가우시안 블러"""
    gray = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2GRAY)
    return cv2.GaussianBlur(gray, BLUR_KERNEL, 0)


def detect_motion(prev_gray: np.ndarray, curr_gray: np.ndarray):
    """
    두 프레임 차이로 움직임 감지.
    반환: (motion_score, thresh_image)
    """
    delta = cv2.absdiff(prev_gray, curr_gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    score = cv2.countNonZero(thresh)
    return score, thresh


def save_image(frame_bgra: np.ndarray, capture_id: str, score: int) -> str:
    """BGR 변환 후 파일 저장, 저장 경로 반환"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"motion_{timestamp}_{capture_id[:8]}.jpg")
    bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(filename, bgr)
    print(f"[저장] score={score:>6}  →  {filename}")
    return filename





def main():
    # ── ZED 카메라 초기화 ──────────────────────────────
    zed = sl.CameraOne()
    init_params = sl.InitParametersOne()
    init_params.camera_resolution = sl.RESOLUTION.HD1200
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"카메라 오픈 실패: {repr(err)}")
        exit(1)

    cam_info = zed.get_camera_information()
    print("\n=== ZED Camera Information ===")
    print(f"  Model     : {cam_info.camera_model}")
    print(f"  Serial    : {cam_info.serial_number}")
    res = cam_info.camera_configuration.resolution
    print(f"  Resolution: {res.width}x{res.height}")
    print(f"  FPS       : {cam_info.camera_configuration.fps}")
    print("==============================\n")
    print(f"움직임 감지 시작... (민감도={SENSITIVITY}, 쿨다운={COOLDOWN_SEC}s)")
    print("종료: q\n")

    image      = sl.Mat()
    prev_gray  = None           # 이전 프레임 (첫 프레임에서 초기화)
    last_saved = 0.0            # 마지막 저장 시각
    key        = -1

    try:
        while key != ord('q'):
            if zed.grab() > sl.ERROR_CODE.SUCCESS:
                key = cv2.waitKey(10)
                continue

            zed.retrieve_image(image)
            frame = image.get_data()          # BGRA numpy array

            curr_gray = preprocess(frame)

            # 첫 프레임은 배경으로만 사용
            if prev_gray is None:
                prev_gray = curr_gray
                continue

            # ── 움직임 감지 ──────────────────────────
            score, thresh = detect_motion(prev_gray, curr_gray)
            now = time.time()

            if score > SENSITIVITY and (now - last_saved) >= COOLDOWN_SEC:
                capture_id = str(uuid.uuid4())
                save_image(frame, capture_id, score)
                send_file('mono', 'rgb', frame, capture_id)
                last_saved = now

            prev_gray = curr_gray   # 프레임 업데이트

            # ── 화면 표시 ─────────────────────────────
            # 원본 미리보기
            display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.putText(display, f"Motion: {score}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255) if score > SENSITIVITY else (0, 255, 0),
                        2)
            cv2.imshow("ZED One - Live", display)

            # 움직임 마스크 미리보기 (보고 싶지 않으면 아래 두 줄 주석 처리)
            thresh_display = cv2.resize(thresh, (640, 360))
            cv2.imshow("Motion Mask", thresh_display)

            key = cv2.waitKey(10)

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("\n카메라 연결 종료.")


if __name__ == "__main__":
    main()