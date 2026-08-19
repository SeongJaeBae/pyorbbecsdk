from pyorbbecsdk import Pipeline, Config, OBSensorType, OBError
import numpy as np
import json


def np_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def find_color_profile(color_profiles, width, height):
    """지원 프로파일 리스트를 순회하며 해당 해상도를 찾음 (fps 가장 높은 것 우선)"""
    candidates = []
    for i in range(len(color_profiles)):
        p = color_profiles[i]
        if p.get_width() == width and p.get_height() == height:
            candidates.append(p)

    if not candidates:
        return None

    # fps 높은 순으로 정렬해서 첫 번째 반환 (원하면 특정 포맷 우선으로 바꿔도 됨)
    candidates.sort(key=lambda p: p.get_fps(), reverse=True)
    return candidates[0]


def intrinsic_to_dict(intr):
    return {
        "fx": intr.fx, "fy": intr.fy,
        "cx": intr.cx, "cy": intr.cy,
        "width": intr.width, "height": intr.height,
    }


def distortion_to_dict(dist):
    return {
        "k1": dist.k1, "k2": dist.k2, "k3": dist.k3,
        "k4": dist.k4, "k5": dist.k5, "k6": dist.k6,
        "p1": dist.p1, "p2": dist.p2,
    }


def get_camera_intrinsics():
    pipeline = Pipeline()
    config = Config()
    result = {}

    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        # 지원되는 컬러 프로파일 전체 출력 (디버깅용)
        print("=== 지원되는 Color 프로파일 ===")
        for i in range(len(color_profiles)):
            p = color_profiles[i]
            print(f"  {p.get_width()}x{p.get_height()} @ {p.get_fps()}fps, format={p.get_format()}")

        # ---- Depth는 기본 프로파일 그대로 ----
        depth_profile = depth_profiles.get_default_video_stream_profile()
        depth_intrinsic = depth_profile.get_intrinsic()
        depth_distortion = depth_profile.get_distortion()

        print("\n=== Depth Intrinsics (default) ===")
        print(f"{depth_intrinsic.width}x{depth_intrinsic.height} "
              f"fx={depth_intrinsic.fx}, fy={depth_intrinsic.fy}, "
              f"cx={depth_intrinsic.cx}, cy={depth_intrinsic.cy}")

        result["depth_intrinsic"] = intrinsic_to_dict(depth_intrinsic)
        result["depth_distortion"] = distortion_to_dict(depth_distortion)

        # ---- Color는 1280x720, 1920x1080 두 개 확인 ----
        target_resolutions = [(1280, 720), (1920, 1080)]
        result["color_intrinsics"] = {}
        result["color_distortions"] = {}
        result["depth_to_color_extrinsic"] = {}

        for width, height in target_resolutions:
            color_profile = find_color_profile(color_profiles, width, height)
            key = f"{width}x{height}"

            if color_profile is None:
                print(f"\n[WARN] Color {key} 프로파일을 찾을 수 없습니다.")
                continue

            color_intrinsic = color_profile.get_intrinsic()
            color_distortion = color_profile.get_distortion()

            print(f"\n=== Color Intrinsics ({key}) @ {color_profile.get_fps()}fps ===")
            print(f"fx: {color_intrinsic.fx}, fy: {color_intrinsic.fy}")
            print(f"cx: {color_intrinsic.cx}, cy: {color_intrinsic.cy}")

            result["color_intrinsics"][key] = intrinsic_to_dict(color_intrinsic)
            result["color_distortions"][key] = distortion_to_dict(color_distortion)

            extrinsic = depth_profile.get_extrinsic_to(color_profile)
            result["depth_to_color_extrinsic"][key] = {
                "rotation": extrinsic.rot,
                "translation": extrinsic.transform,
            }
            print(f"Depth->Color({key}) extrinsic 확인 완료")

        with open("orbbec_calibration.json", "w") as f:
            json.dump(result, f, indent=2, default=np_encoder)

        print("\ncalibration 정보가 orbbec_calibration.json 으로 저장되었습니다.")

    except OBError as e:
        print(f"OBError 발생: {e}")


if __name__ == "__main__":
    get_camera_intrinsics()