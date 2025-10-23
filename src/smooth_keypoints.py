import json
import numpy as np
from scipy.signal import savgol_filter

# Load JSON
with open("../data/black_v4_pose.json") as f:
    pose_data = json.load(f)

# Filter out empty frames
valid_frames = [frame for frame in pose_data if frame and "keypoints" in frame]

num_frames = len(valid_frames)
if num_frames == 0:
    raise ValueError("No valid frames found.")

num_keypoints = len(valid_frames[0]["keypoints"])

# Convert keypoints to NumPy array: shape (num_frames, num_keypoints, 2)
keypoints_array = np.array([frame["keypoints"] for frame in valid_frames], dtype=float)

# Smooth each keypoint over time (axis=0)
# window_length must be odd and <= num_frames
window_length = min(9, num_frames if num_frames % 2 == 1 else num_frames - 1)
polyorder = 2

smoothed_keypoints = savgol_filter(keypoints_array, window_length=window_length, polyorder=polyorder, axis=0)

# Write back to frames
for i, frame in enumerate(valid_frames):
    frame["keypoints"] = smoothed_keypoints[i].tolist()

# Save back to JSON
with open("../data/black_v4_pose_smoothed.json", "w") as f:
    json.dump(pose_data, f, indent=2)
