import json
import numpy as np
from typing import Callable

from src.dsl.summary import SummaryData, HoldUsageSummary, HoldUsageMap

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Process Summary Data"
)

parser.add_argument(
    "--holds_file",
    type=Path,
    help="Path to holds file"
)

parser.add_argument(
    "--pose_file",
    type=Path,
    help="Path to pose file"
)

parser.add_argument(
    "--frame_analytics_file",
    type=Path,
    help="Path to frame analytics file"
)

parser.add_argument(
    "-o",
    "--output",
    type=Path,
    help="Path to output file"
)

args = parser.parse_args()

with open(args.holds_file) as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open(args.pose_file) as pose_f:
    pose_data = json.load(pose_f)

with open(args.frame_analytics_file) as analytics_f:
    analytics_data = json.load(analytics_f)

frame_count = len(pose_data)
fps = 30



def filter_if(ar : np.ndarray, condition : Callable[..., bool]):
    return [x for x in ar if condition(x)]

def count_if(ar : np.ndarray, condition : Callable[..., bool]):
    return np.sum([1 for x in ar if condition(x)])

def round_to_n_digits(number, n : int = 0):
    return round(number * 10 ** n) / 10 ** n


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

summary_data = SummaryData(
    HoldUsageSummary=hold_usage_summary,
    HoldUsageMap=hold_usage_map
)


# Save to JSON
with open(args.output, "w") as f:
    f.write(summary_data.model_dump_json(indent=2))
