import json
import numpy as np
from typing import Callable

with open("../data/blue_v6_holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open("../data/blue_v6_pose_smoothed.json") as pose_f:
    pose_data = json.load(pose_f)

with open("../data/blue_v6_analytics.json") as analytics_f:
    analytics_data = json.load(analytics_f)

frame_count = len(pose_data)
fps = 30

def filter_if(ar : np.ndarray, condition : Callable[..., bool]):
    return [x for x in ar if condition(x)]

def count_if(ar : np.ndarray, condition : Callable[..., bool]):
    return np.sum([1 for x in ar if condition(x)])

def round_to_n_digits(number, n : int = 0):
    return round(number * 10 ** n) / 10 ** n


def time_spent_on_n_contacts(n : int):
    left_hand = analytics_data[53]["data"]
    right_hand = analytics_data[54]["data"]
    left_foot = analytics_data[55]["data"]
    right_foot = analytics_data[56]["data"]

    data = []
    for i in range(frame_count):
        pose = [left_hand[i], right_hand[i], left_foot[i], right_foot[i]]
        data.append(count_if(np.array(pose), lambda x: x is not None) == n)

    return int(count_if(np.array(data), lambda x: x == True)) / fps

summary_data = [
                {"category": "Hold Usage", "name": "4 Contacts", "data": time_spent_on_n_contacts(4)},
                {"category": "Hold Usage", "name": "3 Contacts", "data": time_spent_on_n_contacts(3)},
                {"category": "Hold Usage", "name": "2 Contacts", "data": time_spent_on_n_contacts(2)},
                {"category": "Hold Usage", "name": "1 Contacts", "data": time_spent_on_n_contacts(1)},
                {"category": "Hold Usage", "name": "0 Contacts", "data": time_spent_on_n_contacts(0)}

]

print(summary_data)



# Save to JSON
with open("../data/blue_v6_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)