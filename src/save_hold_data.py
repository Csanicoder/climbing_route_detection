import json
import os.path

import cv2
import numpy as np

from src.hold import inference

import argparse
from pathlib import Path

print("Started running hold recognition!")

parser = argparse.ArgumentParser(
    description="Process Hold Data"
)

parser.add_argument(
    "--route_name",
    type=str,
    help="Common name of route"
)

parser.add_argument(
    "--video_dir",
    type=Path,
    help="Path to directory video is in"
)

parser.add_argument(
    "--data_dir",
    type=Path,
    help="Path to directory output will go to"
)

parser.add_argument(
    "--video_file",
    type=str,
    help="Ending of video file from the route name"
)

parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="Extension of output"
)

args = parser.parse_args()

video_path = os.path.join(args.video_dir, args.route_name + args.video_file)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

ret, frame = cap.read()
if not ret:
    print("No frames!")

#frame = cv2.rotate(frame, cv2.ROTATE_180)

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

#data = [{"centroids": centroids, "boxes": boxes, "classes": classes}]

cut_masks = []

for box, mask in zip(boxes, masks):
    bx = [round(x) for x in box]
    cut_masks.append(mask[bx[1]:bx[3] + 1, bx[0]:bx[2] + 1].tolist()) # slice the pixel mask to the region of the bbox


#data = [{"centroid": centroid, "bbox": [int(round(x)) for x in box], "class": int(c), "cut_mask": cut_mask} for centroid, box, c, cut_mask in zip(centroids, boxes, classes, cut_masks)]
data = [{"centroid": centroid, "bbox": [int(round(x)) for x in box], "class": int(c)} for centroid, box, c in zip(centroids, boxes, classes)]

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
with open(os.path.join(args.data_dir, args.route_name + "_no_mask" + args.output), "w") as f:
    json.dump(data, f, indent=2)

print("Hold data was saved successfully!")