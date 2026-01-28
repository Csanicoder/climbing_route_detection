import json

import numpy as np
from scipy.signal import savgol_filter
# Load Holds JSON file


video_name = "blue_v6"


with open(f"../data/{video_name}_holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file
with open(f"../data/{video_name}_pose_smoothed.json") as pose_f:
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

def body_bbox():
    bboxes = []
    for pose in pose_data:
        if pose and pose.get("keypoints"):
            keypoints = np.array(pose["keypoints"])
            bbox = np.stack((keypoints.min(axis=0), keypoints.max(axis=0)))
            bboxes.append(bbox)
        else:
            bboxes.append(None)  # placeholder for empty poses
    return bboxes

body_bboxes = body_bbox()

def body_center():
    global body_bboxes
    centers = []
    for bbox in body_bboxes:
        if bbox is not None:
            center = np.floor(bbox.mean(axis=0)).astype(int)
            centers.append(center.tolist())  # convert NumPy array to list of ints
        else:
            centers.append(None)  # or [] if you prefer empty list
    return centers

body_centers = body_center()

body_vertical_lines = []    # x_center along full y-range
body_horizontal_lines = []  # y_center along full x-range

def bodycenterlines():
    global body_bboxes
    global body_centers
    global body_vertical_lines
    global body_horizontal_lines

    for bbox, center in zip(body_bboxes, body_centers):
        if bbox is None or center is None:
            body_vertical_lines.append(None)
            body_horizontal_lines.append(None)
            continue

        # Convert to integers for range
        x_min, y_min = map(int, bbox[0])
        x_max, y_max = map(int, bbox[1])
        x_center, y_center = map(int, center)

        # Vertical line: x = x_center, y = y_min -> y_max
        vertical_line = [[x_center, y_min], [x_center, y_max]]
        body_vertical_lines.append(vertical_line)

        # Horizontal line: y = y_center, x = x_min -> x_max
        horizontal_line = [[x_min, y_center], [x_max, y_center]]
        body_horizontal_lines.append(horizontal_line)

bodycenterlines()

# Initialize lists
upper_bboxes, upper_centers, upper_vlines, upper_hlines = [], [], [], []
lower_bboxes, lower_centers, lower_vlines, lower_hlines = [], [], [], []

def upper_and_lower_metrics():
    global pose_data
    global upper_bboxes, upper_centers, upper_vlines, upper_hlines
    global lower_bboxes, lower_centers, lower_vlines, lower_hlines

    for pose in pose_data:
        if not pose or "keypoints" not in pose or not pose["keypoints"]:
            # Append None or empty placeholders
            upper_bboxes.append(None)
            upper_centers.append(None)
            upper_vlines.append(None)
            upper_hlines.append(None)
            lower_bboxes.append(None)
            lower_centers.append(None)
            lower_vlines.append(None)
            lower_hlines.append(None)
            continue

        # Separate upper and lower body points
        upper_points = np.array(pose["keypoints"][5:12])
        lower_points = np.array(pose["keypoints"][11:])

        def compute_metrics(points):
            if points.shape[0] == 0:
                return None, None, None, None
            bbox = np.stack((points.min(axis=0), points.max(axis=0)))  # [min, max]
            center = np.floor(bbox.sum(axis=0) / 2).astype(int)
            x_center, y_center = center

            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[1])

            # Vertical line: x_center along y-range
            vline = np.array([[x_center, y_min], [x_center, y_max]])
            # Horizontal line: y_center along x-range
            hline = np.array([[x_min, y_center], [x_max, y_center]])

            return bbox.tolist(), center.tolist(), vline.tolist(), hline.tolist()

        ubbox, ucenter, uvline, uhline = compute_metrics(upper_points)
        lbbox, lcenter, lvline, lhline = compute_metrics(lower_points)

        # Append metrics to their respective lists
        upper_bboxes.append(ubbox)
        upper_centers.append(ucenter)
        upper_vlines.append(uvline)
        upper_hlines.append(uhline)

        lower_bboxes.append(lbbox)
        lower_centers.append(lcenter)
        lower_vlines.append(lvline)
        lower_hlines.append(lhline)

upper_and_lower_metrics()


left_arm_reach = []
right_arm_reach = []
left_leg_reach = []
right_lef_reach = []
hands_distance = []
left_side_distance = []
fall_diagonal = []
rise_diagonal = []
right_side_distance = []
feet_distance = []

def body_distances():
    global body_centers
    global left_arm_reach, right_arm_reach, left_leg_reach, right_lef_reach
    global hands_distance, left_side_distance, fall_diagonal
    global rise_diagonal, right_side_distance, feet_distance


    # Limb indices
    limb_indices = {
        "left_wrist": 9,
        "right_wrist": 10,
        "left_ankle": 15,
        "right_ankle": 16
    }

    # Limb pairs to compute vectors between
    limb_pairs = [
        ("left_wrist", "right_wrist"),
        ("left_wrist", "left_ankle"),
        ("left_wrist", "right_ankle"),
        ("right_wrist", "left_ankle"),
        ("right_wrist", "right_ankle"),
        ("left_ankle", "right_ankle")
    ]

    # Mapping for easier loop assignment
    center_lists = {
        "left_wrist": left_arm_reach,
        "right_wrist": right_arm_reach,
        "left_ankle": left_leg_reach,
        "right_ankle": right_lef_reach
    }

    pair_lists = {
        ("left_wrist", "right_wrist"): hands_distance,
        ("left_wrist", "left_ankle"): left_side_distance,
        ("left_wrist", "right_ankle"): fall_diagonal,
        ("right_wrist", "left_ankle"): rise_diagonal,
        ("right_wrist", "right_ankle"): right_side_distance,
        ("left_ankle", "right_ankle"): feet_distance
    }

    # Compute vectors
    for idx, pose in enumerate(pose_data):
        if not pose or "keypoints" not in pose or not pose["keypoints"]:
            # Append None for missing poses
            for lst in center_lists.values():
                lst.append(None)
            for lst in pair_lists.values():
                lst.append(None)
            continue

        keypoints = np.array(pose["keypoints"])
        body_center = np.array(body_centers[idx])

        # Vectors from center to each limb
        limb_coords = {}
        for name, limb_idx in limb_indices.items():
            point = keypoints[limb_idx]
            limb_coords[name] = point
            vector = point - body_center
            magnitude = float(np.linalg.norm(vector))
            center_lists[name].append((vector.tolist(), magnitude))

        # Vectors between limb pairs
        for pair, lst in pair_lists.items():
            v = limb_coords[pair[1]] - limb_coords[pair[0]]
            mag = float(np.linalg.norm(v))
            line = [limb_coords[pair[0]].tolist(), limb_coords[pair[1]].tolist()]
            lst.append((line, mag))

body_distances()

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

def com_distance_to_body_center():
    data = []
    global com_positions
    global body_centers

    for com, center in zip(com_positions.tolist(), body_centers):
        if all(a == 0 for a in com) or center is None:
            data.append(None)
            continue

        data.append(([com, center], np.linalg.norm(np.array(com) - np.array(center)).tolist()))
    return data

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

def unpack(a_list):
    out = []

    item_identifier = 0
    number_of_items = 1

    for item in a_list:
        out += [item[item_identifier]] * item[number_of_items]

    return out

def block_fill(a_list, error_threshold : int):

    item_identifier = 0
    number_of_items = 1

    packed_list = pack(a_list)
    for i, item in enumerate(packed_list[1:len(packed_list) - 1]): #correct error by filling in short -1-s between data
        left_item_identifier = packed_list[i][item_identifier]
        right_item_identifier = packed_list[i + 2][item_identifier]

        if (left_item_identifier == right_item_identifier != -1) and item[number_of_items] <= error_threshold:
            packed_list[i + 1] = (left_item_identifier, item[number_of_items])

    packed_list = pack(unpack(packed_list))

    for i, item in enumerate(packed_list[1:len(packed_list) - 1]): #correct error by erasing in short data between -1-s

        if item[number_of_items] <= error_threshold:
            packed_list[i + 1] = (-1, item[number_of_items])

    return unpack(packed_list)

def wall_contact(limb_index):
    global hold_data
    global pose_data
    global keypoint_velocities

    limb_keypoint_map = {0: 9, 1: 10, 2: 15, 3: 16}

    keypoint_index = limb_keypoint_map[limb_index]

    data = []

    for pose in pose_data:

        i = pose_data.index(pose)

        if not pose:
            data.append(-1)
            continue

        for hold in hold_data:
            keypoint_pos = pose["keypoints"][keypoint_index]

            keypoint_vel = keypoint_velocities[keypoint_index - 5][i][1]
            hold_bbox = hold["bbox"]

            if hold_bbox[0] < keypoint_pos[0] < hold_bbox[2] and hold_bbox[1] < keypoint_pos[1] < hold_bbox[3]:
                data.append(hold_data.index(hold))
                break

        if len(data) <= i: #if the limb doesn't hold anything, append -1
            data.append(-1)

    data = block_fill(data, 20)


    return data




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

    {"category": "Body Layout", "name": "Body Bbox", "data": body_bboxes},
    {"category": "Body Layout", "name": "Body Center", "data": body_centers},
    {"category": "Body Layout", "name": "Horizontal Body Centerline", "data": body_horizontal_lines},
    {"category": "Body Layout", "name": "Vertical Body Centerline", "data": body_vertical_lines},
    {"category": "Body Layout", "name": "Upper Body Bbox", "data": upper_bboxes},
    {"category": "Body Layout", "name": "Upper Body Center", "data": upper_centers},
    {"category": "Body Layout", "name": "Upper Horizontal Body Centerline", "data": upper_hlines},
    {"category": "Body Layout", "name": "Upper Vertical Body Centerline", "data": upper_vlines},
    {"category": "Body Layout", "name": "Lower Body Bbox", "data": lower_bboxes},
    {"category": "Body Layout", "name": "Lower Body Center", "data": lower_centers},
    {"category": "Body Layout", "name": "Lower Horizontal Body Centerline", "data": lower_hlines},
    {"category": "Body Layout", "name": "Lower Vertical Body Centerline", "data": lower_vlines},

    {"category": "Body Distances", "name": "Left Arm Reach", "data": left_arm_reach},
    {"category": "Body Distances", "name": "Right Arm Reach", "data": right_arm_reach},
    {"category": "Body Distances", "name": "Left Leg Reach", "data": left_leg_reach},
    {"category": "Body Distances", "name": "Right Leg Reach", "data": right_lef_reach},
    {"category": "Body Distances", "name": "Hands Distance", "data": hands_distance},
    {"category": "Body Distances", "name": "Feet Distance", "data": feet_distance},
    {"category": "Body Distances", "name": "Left Side Distance", "data": left_side_distance},
    {"category": "Body Distances", "name": "Right Side Distance", "data": right_side_distance},
    {"category": "Body Distances", "name": "Fall Diagonal", "data": fall_diagonal},
    {"category": "Body Distances", "name": "Rise Diagonal", "data": rise_diagonal},

    {"category": "Center of Mass", "name": "Position", "data": com_positions.tolist()},
    {"category": "Center of Mass", "name": "Velocity", "data": com_combined_vel},
    {"category": "Center of Mass", "name": "Distance to Body Center", "data": com_distance_to_body_center()}]

for i in range(8):
    analytics_data.append({"category": "Joint Angles", "name": joint_names[i], "data": joint_angles[i]})

for i in range(8):
    analytics_data.append({"category": "Joint Angular Velocities", "name": joint_names[i], "data": joint_angular_vel[i]})

for i in range(12):
    analytics_data.append({"category": "Keypoint Velocities", "name": keypoint_names[i], "data": keypoint_velocities[i]})

analytics_data.append({"category": "Wall Contacts", "name": "Left Hand", "data": wall_contact(0)})
analytics_data.append({"category": "Wall Contacts", "name": "Right Hand", "data": wall_contact(1)})
analytics_data.append({"category": "Wall Contacts", "name": "Left Foot", "data": wall_contact(2)})
analytics_data.append({"category": "Wall Contacts", "name": "Right Foot", "data": wall_contact(3)})


# Convert all NumPy arrays in your data to lists
def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    return obj

# Save to JSON
with open(f"../data/{video_name}_analytics.json", "w") as f:
    json.dump(convert_ndarray_to_list(analytics_data), f, indent=2)