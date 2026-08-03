from pyorbbecsdk import *

pipeline = Pipeline()
config = Config()

color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()

config.enable_stream(color_profile)
config.enable_stream(depth_profile)

profile = pipeline.start(config)

color_stream_profile = color_profile.as_video_stream_profile()
depth_stream_profile = depth_profile.as_video_stream_profile()

rgb_intr = color_stream_profile.get_intrinsic()
rgb_dist = color_stream_profile.get_distortion()

depth_intr = depth_stream_profile.get_intrinsic()
depth_dist = depth_stream_profile.get_distortion()

print("RGB")
print("fx:", rgb_intr.fx)
print("fy:", rgb_intr.fy)
print("cx:", rgb_intr.cx)
print("cy:", rgb_intr.cy)
print("width:", color_stream_profile.get_width())
print("height:", color_stream_profile.get_height())
print("distortion:", rgb_dist)

print("DEPTH")
print("fx:", depth_intr.fx)
print("fy:", depth_intr.fy)
print("cx:", depth_intr.cx)
print("cy:", depth_intr.cy)
print("width:", depth_stream_profile.get_width())
print("height:", depth_stream_profile.get_height())
print("distortion:", depth_dist)

pipeline.stop()