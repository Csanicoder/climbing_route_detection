import json
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter

# Load Holds JSON file
with open("data/holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open("data/pose_smoothed.json") as pose_f:
    pose_data = json.load(pose_f)

frame_count = len(pose_data)
print(frame_count)

fps = 30

# Here we define the functions that create the analytics elements

def differentiate(values : np.ndarray, delta_t : float, axis : int):
    num_frames = values.shape[axis]
    t = np.arange(num_frames) * delta_t

    return np.gradient(values, t, axis=axis)

def smooth_differentiate(values : np.ndarray, delta_t : float, axis : int):
    return savgol_filter(values, window_length=5, polyorder=3, deriv=1, delta=delta_t, axis=axis)

def speed_from_velocity(velocities : np.ndarray):
    return np.linalg.norm(velocities, axis=1)

def filter_if(ar : np.ndarray, condition : Callable[..., bool]):
    return [x for x in ar if condition(x)]

def count_if(ar : np.ndarray, condition : Callable[..., bool]):
    return np.sum([1 for x in ar if condition(x)])


'''def calculate_point_velocity(positions : np.array):
    data = []
    for i in range(len(positions)):
        values = [] # Item 0 of tuple is the velocity vector, the second is the magnitude

        pos1 = max(0, i - 1) # Bottom clamp
        pos2 = min(len(positions) - 1, i + 1) # Top clamp
        delta = pos2 - pos1 # Get the number of frames we divide over

        displacement = positions[pos2] - positions[pos1] # displacement vector
        displacement_m = sqrt(pow(displacement[0], 2) + pow(displacement[1], 2)) # displacement magnitude, pythagorean theorem
        timeframe = delta * dt # time we divide over in seconds

        values.append((displacement / timeframe).tolist())
        values.append(displacement_m / timeframe)

        data.append(values)

    return data
'''

com_positions = np.array(None)
angles_keypoints_map = [
    [11, 5, 7], # left armpit
    [12, 6, 8], # right armpit
    [5, 7, 9], # left elbow
    [6, 8, 10], # right elbow
    [5, 11, 13], # left hip (outside angle)
    [6, 12, 14], # right hip (outside angle)
    [11, 13, 15], # left knee
    [12, 14, 16] # right knee
]
keypoint_velocities = []

def com_pos():
    data = []
    global com_positions

    for pose in pose_data: # loop through each frame
        if not pose: # no keypoints on the frame
            data.append([0, 0])
            continue

        # Helper: average of two or more points
        def avg(*pts):
            return np.mean(np.array(pts), axis=0)

        keypoints = pose["keypoints"]

        # Define segment midpoints
        segments = [
            ("head", avg(keypoints[0], avg(keypoints[3], keypoints[4]))),  # nose + ears
            ("torso", avg(avg(keypoints[5], keypoints[6]), avg(keypoints[11], keypoints[12]))),
            ("L_upper_arm", avg(keypoints[5], keypoints[7])),
            ("R_upper_arm", avg(keypoints[6], keypoints[8])),
            ("L_forearm", avg(keypoints[7], keypoints[9])),
            ("R_forearm", avg(keypoints[8], keypoints[10])),
            ("L_thigh", avg(keypoints[11], keypoints[13])),
            ("R_thigh", avg(keypoints[12], keypoints[14])),
            ("L_shank", avg(keypoints[13], keypoints[15])),
            ("R_shank", avg(keypoints[14], keypoints[16])),
        ]

        # Weights (% of body mass)
        weights = np.array([8, 50, 3, 3, 2, 2, 10, 10, 5, 5], dtype=float)
        weights /= weights.sum()  # normalize to sum=1

        seg_points = np.array([p for _, p in segments]) # get only the positions of the segments
        x_com = np.sum(seg_points[:, 0] * weights)
        y_com = np.sum(seg_points[:, 1] * weights)

        data.append((x_com, y_com)) # add average keypoint pos to list

    com_positions = savgol_filter(np.array(data), window_length=9, polyorder=2, axis=0)
def calc_joint_angles():
    global com_positions

    data = [
        [],[],[],[],[],[],[],[]
    ]

    for pose in pose_data:
        if not pose:
            for i in range(len(data)):
                data[i].append(0)
            continue

        keypoints = pose["keypoints"]

        for angle_map in angles_keypoints_map:
            u = np.array(keypoints[angle_map[1]]) - np.array(keypoints[angle_map[0]])
            v = np.array(keypoints[angle_map[1]]) - np.array(keypoints[angle_map[2]])

            cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # prevent NaNs
            theta = np.arccos(cos_theta)
            theta_deg = np.degrees(theta)

            data[angles_keypoints_map.index(angle_map)].append(theta_deg)

    return np.array(data)
def calc_joint_angular_vel():
    global joint_angles
    global frame_count

    data = []

    for angles in joint_angles:

        data.append(smooth_differentiate(angles, 1 / fps, axis=0).tolist())

    return data
def calculate_keypoint_velocities(): # calculate velocities for each keypoint

    # restructure keypoint positions, leave out head to avoid clutter
    keypoint_positions = [[] for _ in range(12)]
    for pose in pose_data: # if no pose data, append 0 to positions

        if not pose:
            for pos in keypoint_positions:
                pos.append([0,0])
            continue

        for i in range(5, 17): # loop through keypoints in each frame, and append them to the corresponding list
            keypoint_positions[i - 5].append(pose["keypoints"][i])


    for i in range(5, 17): # use the previous velocity function on each keypoint position list
        vel = smooth_differentiate(np.array(keypoint_positions[i - 5]), 1 / fps, 0).tolist()
        speed = np.stack([speed_from_velocity(vel)]*2, axis=1).tolist()
        combined_vel = [[v.tolist(), s[0]] for v, s in np.stack([vel, speed], axis=1)]
        keypoint_velocities.append(combined_vel)


com_pos()
com_vel = smooth_differentiate(com_positions, 1 / fps, 0)
com_speed = np.stack([speed_from_velocity(com_vel)]*2, axis=1)
com_combined_vel = [[v.tolist(), s[0]] for v, s in np.stack([com_vel, com_speed], axis=1)]

joint_angles = calc_joint_angles().tolist()

joint_angular_vel = calc_joint_angular_vel()

calculate_keypoint_velocities()



joint_names = ["Left Armpit", "Right Armpit", "Left Elbow", "Right Elbow", "Left Hip", "Right Hip", "Left Knee", "Right Knee"]
keypoint_names = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist",
                  "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]


analytics_data = [

    {"category": "Center of Mass", "name": "Position", "data": com_positions.tolist()},
    {"category": "Center of Mass", "name": "Velocity", "data": com_combined_vel}]



for i in range(8):
    analytics_data.append({"category": "Joint Angles", "name": joint_names[i], "data": joint_angles[i]})

for i in range(8):
    analytics_data.append({"category": "Joint Angular Velocities", "name": joint_names[i], "data": joint_angular_vel[i]})

for i in range(12):
    analytics_data.append({"category": "Keypoint Velocities", "name": keypoint_names[i], "data": keypoint_velocities[i]})


# Save to JSON
with open("data/analytics.json", "w") as f:
    json.dump(analytics_data, f, indent=2)