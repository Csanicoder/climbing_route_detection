import json
import cv2
import DataVisualizer

# Load Holds JSON file
with open("holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open("pose.json") as pose_f:
    pose_data = json.load(pose_f)

video_path = "slab.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# opening the writing to the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))

dv = DataVisualizer.DataVisualizer()

for i in range(frame_count):

    print(f"Processing frame {i}")

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = dv.visualize(hold_data[i], pose_data[i], frame_rgb)

    save_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out.write(save_bgr)

cap.release()
out.release()

