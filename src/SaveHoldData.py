import json
import cv2

from src.hold import inference

video_path = "slab.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

data = []

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

for i in range(frame_count):
    print(f"Reading frame {i}")
    ret, frame = cap.read()
    if not ret:
        print("No more frames!")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    print(f"Proccessing frame {i}")

    hold_instances = inference.inference(frame)["instances"]    # Run inference on the frame

    print("Preparing frame data")

    boxes = hold_instances.pred_boxes.tensor.cpu().tolist()  # convert tensor to list
    classes = hold_instances.pred_classes.cpu().tolist()     # convert tensor to list

    # Prepare data for JSON
    frame_data = [{"bbox": [int(round(x)) for x in box], "class": int(c)}
                 for box, c in zip(boxes, classes)]

    print("Appending frame data")

    data.append(frame_data)

    print(f"Proccessed frame {i}")

cap.release()

# Save to JSON
with open("holds.json", "w") as f:
    json.dump(data, f, indent=2)