import json
import os.path

import numpy as np
from typing import Callable

from src.dsl.summary import SummaryData, HoldUsageSummary, HoldUsageMap, RouteSegmentationItem

import argparse
from pathlib import Path

print("Started running summary generation!")

parser = argparse.ArgumentParser(
    description="Process Summary Data"
)

parser.add_argument(
    "--route_name",
    type=str,
    help="Common name of route"
)

parser.add_argument(
    "--data_dir",
    type=Path,
    help="Path to directory output will go to"
)

parser.add_argument(
    "--holds_file",
    type=str,
    help="Ending of holds file from the route name"
)

parser.add_argument(
    "--pose_file",
    type=str,
    help="Ending of pose file from the route name"
)

parser.add_argument(
    "--analytics_file",
    type=str,
    help="Ending of analytics file from the route name"
)

parser.add_argument(
    "--output",
    "-o",
    type=str,
    help="Extension of output"
)

args = parser.parse_args()

with open(os.path.join(args.data_dir, args.route_name + args.holds_file)) as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open(os.path.join(args.data_dir, args.route_name + args.pose_file)) as pose_f:
    pose_data = json.load(pose_f)

with open(os.path.join(args.data_dir, args.route_name + args.analytics_file)) as analytics_f:
    analytics_data = json.load(analytics_f)

frame_count = len(pose_data)
fps = 30



def filter_if(ar : np.ndarray, condition : Callable[..., bool]):
    return [x for x in ar if condition(x)]

def count_if(ar : np.ndarray, condition : Callable[..., bool]):
    return np.sum([1 for x in ar if condition(x)])

def round_to_n_digits(number, n : int = 0):
    return round(number * 10 ** n) / 10 ** n

def pack(a_list):
    out = []
    count = 1
    prev = a_list[0]

    for item in a_list[1:]:
        if item != prev:
            out.append((prev, count))
            count = 1
            prev = item
        else:
            count += 1

    out.append((prev, count))
    return out


def time_spent_on_n_contacts(n : int) -> float:
    left_hand = analytics_data[53]["data"]
    right_hand = analytics_data[54]["data"]
    left_foot = analytics_data[55]["data"]
    right_foot = analytics_data[56]["data"]

    data = []
    for i in range(frame_count):
        pose = [left_hand[i], right_hand[i], left_foot[i], right_foot[i]]
        data.append(count_if(np.array(pose), lambda x: x >= 0) == n)

    answer = int(count_if(np.array(data), lambda x: x == True)) / fps
    return round_to_n_digits(answer, 3)


hold_usage_summary = HoldUsageSummary(
    limb0=time_spent_on_n_contacts(0),
    limb1=time_spent_on_n_contacts(1),
    limb2=time_spent_on_n_contacts(2),
    limb3=time_spent_on_n_contacts(3),
    limb4=time_spent_on_n_contacts(4)
)

hold_usage_map = HoldUsageMap(
    left_hand=analytics_data[53]["data"],
    right_hand = analytics_data[54]["data"],
    left_foot = analytics_data[55]["data"],
    right_foot = analytics_data[56]["data"]
)

list_a = pack(analytics_data[53]["data"]) # left hand
list_b = pack(analytics_data[54]["data"]) # right hand

route_seg = []

# left hand
frame_index = list_a[0][1]
for i in range(1, len(list_a) - 1):

    item = list_a[i]
    prev_item_id = list_a[i - 1][0] # id of previous hold
    next_item_id = list_a[i + 1][0] # id of next hold

    if item[0] == -1:
        route_seg.append((frame_index,
                          frame_index + item[1],
                          0,
                          prev_item_id,
                          next_item_id))
    frame_index += item[1]


 # right hand
frame_index = list_b[0][1]
for i in range(1, len(list_b) - 1):

    item = list_b[i]
    prev_item_id = list_b[i - 1][0] # id of previous hold
    next_item_id = list_b[i + 1][0] # id of next hold

    if item[0] == -1:
        route_seg.append((frame_index,
                          frame_index + item[1],
                          1,
                          prev_item_id,
                          next_item_id))
    frame_index += item[1]



route_seg.sort()

frame_move_map = [-1] * route_seg[0][0] # fill -1 until first move


for index, move in enumerate(route_seg[:len(route_seg) - 1]): # enumerate up to the last move (exclusive)
    move_length = move[1] - move[0] + 1 # the number of frames to append for each move
    frame_move_map += [index] * move_length

    rest_length = route_seg[index + 1][0] - move[1] - 1 # the number of frames between this and the next move
    frame_move_map += [-1 * (index + 2)] * rest_length

# finish off with the last move and the last rest
last_move = route_seg[len(route_seg) - 1]
last_move_length = last_move[1] - last_move[0] + 1 # the number of frames to append for last move
frame_move_map += [len(route_seg) - 1] * last_move_length # append the last index

last_rest_length = frame_count - last_move[1] - 1
frame_move_map += [-1 * (len(route_seg) + 1)] * last_rest_length



route_seg_final = []

for move in route_seg:
    route_seg_final.append(
        RouteSegmentationItem(
            start_frame=move[0],
            end_frame=move[1],
            limb_index=move[2],
            hold_from=move[3],
            hold_to=move[4]
        )
    )



summary_data = SummaryData(
    HoldUsageSummary=hold_usage_summary,
    HoldUsageMap=hold_usage_map,
    RouteSegmentation=route_seg_final,
    FrameMoveMap=frame_move_map
)


# Save to JSON
with open(os.path.join(args.data_dir, args.route_name + args.output), "w") as f:
    f.write(summary_data.model_dump_json(indent=2))

print("Summary data saved successfully!")