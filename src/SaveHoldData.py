import json
import cv2
import numpy as np

from src.hold import inference

video_path = "../video/black_v4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

ret, frame = cap.read()
if not ret:
    print("No frames!")

frame = cv2.rotate(frame, cv2.ROTATE_180)

hold_instances = inference.inference(frame)["instances"]

masks = hold_instances.pred_masks.cpu().tolist()
masks = np.array(masks)
centroids = []
boxes = hold_instances.pred_boxes.tensor.cpu().tolist()  # convert tensor to list
classes = hold_instances.pred_classes.cpu().tolist()

for mask in masks:
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError("No True pixels found")

    else:
        centroid_x = int(round(xs.mean()))
        centroid_y = int(round(ys.mean()))

    centroid = (centroid_x, centroid_y)
    centroids.append(centroid)

data = [{"centroids": centroids, "boxes": boxes, "classes": classes}]

data = [{"centroid": centroid, "bbox": [int(round(x)) for x in box], "class": int(c)} for centroid, box, c in zip(centroids, boxes, classes)]

print(data)

'''
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

'''

cap.release()

# Save to JSON
with open("../data/black_v4_holds.json", "w") as f:
    json.dump(data, f, indent=2)