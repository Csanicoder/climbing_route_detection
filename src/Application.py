import os.path
import time
from math import sqrt
import json

import dearpygui.dearpygui as dpg
import cv2
import numpy as np

import VideoClass
import plots.AbstractPlot

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Process Summary Data"
)

parser.add_argument(
    "--route_name",
    type=str,
    help="Common name of route"
)

parser.add_argument(
    "--video_dir",
    type=Path,
    help="Path to directory output will go to"
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
    "--summary_file",
    type=str,
    help="Ending of summary file from the route name"
)

args = parser.parse_args()

Video = VideoClass.Video((os.path.join(args.video_dir, args.route_name + args.video_file).__str__()))

# Load Holds JSON file with hold data
with open(os.path.join(args.data_dir, args.route_name + args.holds_file)) as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file with pose data
with open(os.path.join(args.data_dir, args.route_name + args.pose_file)) as pose_f:
    pose_data = json.load(pose_f)

# Load Pose JSON file with analytics
with open(os.path.join(args.data_dir, args.route_name + args.analytics_file)) as analytics_f:
    analytics_data = json.load(analytics_f)

with open(os.path.join(args.data_dir, args.route_name + args.summary_file)) as summary_f:
    summary_data = json.load(summary_f)




viewport_width = 1920
viewport_height = 1080

texture_height = viewport_height - 100
texture_width = int((Video.WIDTH / Video.HEIGHT) * texture_height)

PLAYBACK_FPS = Video.FPS

annotation_coefficient = Video.WIDTH / 1080

# Create DearPyGui context
dpg.create_context()

with dpg.font_registry():
    font = dpg.add_font("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

def create_video_texture():
    """Create the texture entry for displaying the video"""

    # Texture registry
    with dpg.texture_registry():
        frame = cv2.cvtColor(Video.current(), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (texture_width, texture_height))

        texture_data = frame.astype(np.float32).flatten() / 255.0

        dpg.add_raw_texture(texture_width, texture_height, texture_data,
                            format=dpg.mvFormat_Float_rgb,
                            tag="video_texture")

create_video_texture()


visualize_elements = [False] * len(analytics_data)
refresh_visualize : bool = True
do_visualize_keypoints : bool = False
do_visualize_bones : bool = False
do_visualize_holds : bool = False
do_display_move_index : bool = False
do_display_current_move : bool = False
isVideoPaused = False
current_move_index = 0

def set_visualize_keypoints(sender):
    global do_visualize_keypoints
    do_visualize_keypoints = dpg.get_value(sender)

def set_visualize_bones(sender):
    global do_visualize_bones
    do_visualize_bones = dpg.get_value(sender)

def set_visualize_holds(sender):
    global do_visualize_holds
    do_visualize_holds = dpg.get_value(sender)

def set_display_move_index(sender):
    global do_display_move_index
    do_display_move_index = dpg.get_value(sender)

def set_display_current_move(sender):
    global do_display_current_move
    do_display_current_move = dpg.get_value(sender)

def visualize_keypoints(frame_index, rgb_image):
    global pose_data

    if not pose_data[frame_index]:
        return rgb_image

    result = rgb_image

    for keypoint in pose_data[frame_index]["keypoints"]:
        if pose_data[frame_index]["keypoints"].index(keypoint) in [0, 1, 2, 3, 4]:
            continue
        result = cv2.circle(rgb_image, [int(a) for a in keypoint], radius=round(8 * annotation_coefficient), color=(0, 255, 255), thickness=-1)

    return result

def visualize_bones(frame_index, rgb_image):
    global pose_data

    if not pose_data[frame_index]:
        return rgb_image

    result = rgb_image

    connections = [(1, 2), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
                   (13, 15), (12, 14), (14, 16)]

    for bone in connections:
        keypoints = pose_data[frame_index]["keypoints"]
        keypoints[bone[0]] = [int(x) for x in keypoints[bone[0]]]
        keypoints[bone[1]] = [int(x) for x in keypoints[bone[1]]]
        result = cv2.line(rgb_image, keypoints[bone[0]], keypoints[bone[1]], (255, 255, 0), thickness=round(6 * annotation_coefficient))

    return result

def visualize_holds(rgb_image):
    global hold_data

    result = rgb_image

    for hold in hold_data:
        p1 = hold["bbox"][:2]
        p2 = hold["bbox"][2:]
        result = cv2.rectangle(rgb_image, [int(a) for a in p1], [int(a) for a in p2], color=(31, 56, 158), thickness=round(4 * annotation_coefficient))

    return result

def display_move(move_idx, rgb_image):
    global summary_data
    global hold_data

    color = np.array((240, 120, 120))
    in_move = True

    if move_idx < 0:
        in_move = False
        color = np.array((130, 120, 190))
        move_idx = abs(move_idx) - 1

    if move_idx >= len(summary_data["RouteSegmentation"]):
        return rgb_image

    hold_from_idx = summary_data["RouteSegmentation"][move_idx]["hold_from"]
    hold_to_idx = summary_data["RouteSegmentation"][move_idx]["hold_to"]

    result = cv2.rectangle(rgb_image, hold_data[hold_from_idx]["bbox"][0:2], hold_data[hold_from_idx]["bbox"][2:4], color=color * 0.7, thickness=round(6 * annotation_coefficient))
    cv2.rectangle(result, hold_data[hold_to_idx]["bbox"][0:2], hold_data[hold_to_idx]["bbox"][2:4], color=color * 1.1, thickness=round(6 * annotation_coefficient))

    if in_move:
        cv2.line(result, hold_data[hold_from_idx]["centroid"], hold_data[hold_to_idx]["centroid"], color=tuple(int(c) for c in color * 1.5), thickness=round(10*annotation_coefficient))

    return result

def display_analytics_panel(frame_index):
    for i, element in enumerate(analytics_data[12:53]):
        display = 0

        j = i + 12

        if 12 <= j <= 15: # reach
            if analytics_data[j]["data"][frame_index] is None:
                display = 0
            else:
                display = analytics_data[j]["data"][frame_index][1] / 400

        elif 16 <= j <= 21: # distances
            if analytics_data[j]["data"][frame_index] is None:
                display = 0
            else:
                display = analytics_data[j]["data"][frame_index][1] / 700

        elif j == 23 or 52 >= j >= 41: # Velocity of Center of Mass, keypoint velocities
            display = analytics_data[j]["data"][frame_index][1] # magnitude of velocity vector, divide by arbitrary max velocity
            if j == 23:
                display /= 500
            else:
                display /= 1200

        elif j == 24:
            if not analytics_data[j]["data"][frame_index]:
                continue
            display = analytics_data[j]["data"][frame_index][1] / 500

        elif 25 <= j <= 32:
            display = analytics_data[j]["data"][frame_index] / 180

        elif 33 <= j <= 40:
            display = max(0, analytics_data[j]["data"][frame_index] / 2000 + 0.5)

        dpg.set_value("e" + str(j), display)

        if 12 <= j <= 32: # joints, reaches, the more extended they are, the better
            if display < 0.3:
                dpg.bind_item_theme("e" + str(j), red_theme)
            elif display < 0.7:
                dpg.bind_item_theme("e" + str(j), yellow_theme)
            else:
                dpg.bind_item_theme("e" + str(j), green_theme)
        else: # velocities, the slower they are, the better
            if display < 0.3:
                dpg.bind_item_theme("e" + str(j), green_theme)
            elif display < 0.7:
                dpg.bind_item_theme("e" + str(j), yellow_theme)
            else:
                dpg.bind_item_theme("e" + str(j), red_theme)

def visualize_element(sender, app_data): # callback for setting the display of elements
    visualize_elements[int(sender[1:])] = app_data

def visualize(element_id, rgb_image, frame_index):

    if element_id % 4 == 0 and element_id <= 8:
        bbox = analytics_data[element_id]["data"][frame_index]
        if not bbox:
            return rgb_image
        if element_id == 0:
            color = (0, 165, 255)
        elif element_id == 4:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        result = cv2.rectangle(rgb_image, [int(a) for a in bbox[0]], [int(a) for a in bbox[1]], color=color, thickness=round(8 * annotation_coefficient))
        return result

    elif element_id % 4 == 1 and element_id <= 9:
        body_center = analytics_data[element_id]["data"][frame_index]
        if not body_center:
            return rgb_image
        if element_id == 1:
            color = (0, 165, 255)
        elif element_id == 5:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)
        result = cv2.circle(rgb_image, [int(a) for a in body_center], radius=round(12 * annotation_coefficient), color=color, thickness=-1)
        return result


    elif 2 <= element_id % 4 <= 3 and element_id <= 11:
        line = analytics_data[element_id]["data"][frame_index]
        if not line:
            return rgb_image
        if element_id <= 3:
            color = (180, 220, 255)
        elif element_id <= 7:
            color = (200, 255, 200)
        else:
            color = (255, 255, 200)
        result = cv2.line(rgb_image, [int(a) for a in line[0]], [int(a) for a in line[1]], color=color, thickness=round(6 * annotation_coefficient))
        return result

    elif element_id <= 15: #reach
        if not analytics_data[element_id]["data"][frame_index]:
            return rgb_image

        v = analytics_data[element_id]["data"][frame_index][0]

        body_center = analytics_data[1]["data"][frame_index]

        display = analytics_data[element_id]["data"][frame_index][1] / 400

        if dpg.get_value("tb" + analytics_data[element_id]["category"]):
            if display >= 0.3:
                return rgb_image

        if display < 0.3:
            color = (100, 100, 255) # red
        elif display < 0.7:
            color = (120, 220, 235) # yellow
        else:
            color = (140, 215, 140) # green

        p2 = [int(a + b) for a, b in zip(body_center, v)]
        result = cv2.line(rgb_image, [int(a) for a in body_center], p2, color=color, thickness=round(6 * annotation_coefficient))
        return result

    elif element_id <= 21: # distances
        if not analytics_data[element_id]["data"][frame_index]:
            return rgb_image
        line = analytics_data[element_id]["data"][frame_index][0]

        display = analytics_data[element_id]["data"][frame_index][1] / 700

        if dpg.get_value("tb" + analytics_data[element_id]["category"]):
            if display >= 0.3:
                return rgb_image

        if display < 0.3:
            color = (100, 100, 255) # red
        elif display < 0.7:
            color = (120, 220, 235) # yellow
        else:
            color = (140, 215, 140) # green

        result = cv2.line(rgb_image, [int(a) for a in line[0]], [int(a) for a in line[1]], color=color, thickness=round(6 * annotation_coefficient))
        return result

    # Position of Center of Mass
    elif element_id == 22:
        com_pos = analytics_data[22]["data"][frame_index]
        com_pos[0] = int(com_pos[0])
        com_pos[1] = int(com_pos[1])
        result = cv2.circle(rgb_image, com_pos, radius=round(12 * annotation_coefficient), color=(0, 0, 255), thickness=-1)
        return result


    # Velocity of Center of Mass
    elif element_id == 23:
        com_pos = analytics_data[22]["data"][frame_index]
        com_pos[0] = int(com_pos[0])
        com_pos[1] = int(com_pos[1])
        xv, yv  = analytics_data[23]["data"][frame_index][0] # the velocity vector
        com_vel = [int(xv * 0.2), int(yv * 0.2)]
        result = cv2.line(rgb_image, com_pos, (np.array(com_pos) + np.array(com_vel)).tolist(), thickness=round(8 * annotation_coefficient), color=(0, 255, 0))
        result = cv2.circle(result, (np.array(com_pos) + np.array(com_vel)).tolist(), radius=round(8 * annotation_coefficient), color=(0, 255, 0), thickness=-1)
        return result

    elif element_id == 24: # distance to body center
        if not analytics_data[element_id]["data"][frame_index]:
            return rgb_image
        line = analytics_data[element_id]["data"][frame_index][0]
        if not line:
            return rgb_image
        result = cv2.line(rgb_image, [int(a) for a in line[0]], [int(a) for a in line[1]], color=(40, 100, 255), thickness=round(10 * annotation_coefficient))
        return result

    elif 25 <= element_id <= 32: # joint angles
        if not pose_data[frame_index]:
            return rgb_image

        angle_val = analytics_data[element_id]["data"][frame_index] / 180

        if dpg.get_value("tb" + analytics_data[element_id]["category"]):
            if angle_val >= 0.3:
                return rgb_image

        angles_keypoints_map = [
            [11, 5, 7],  # left armpit
            [12, 6, 8],  # right armpit
            [5, 7, 9],  # left elbow
            [6, 8, 10],  # right elbow
            [5, 11, 13],  # left hip (outside angle)
            [6, 12, 14],  # right hip (outside angle)
            [11, 13, 15],  # left knee
            [12, 14, 16]  # right knee
        ]

        joint_idx = angles_keypoints_map[element_id - 25][1]
        src_joint_idx = angles_keypoints_map[element_id - 25][0]
        dest_joint_idx = angles_keypoints_map[element_id - 25][2]

        joint_pos = np.array(pose_data[frame_index]["keypoints"][joint_idx])
        src_joint_pos = np.array(pose_data[frame_index]["keypoints"][src_joint_idx])
        dest_joint_pos = np.array(pose_data[frame_index]["keypoints"][dest_joint_idx])

        src_vec = src_joint_pos - joint_pos
        dest_vec = dest_joint_pos - joint_pos

        # normalize and then scale by 100
        src_vec = src_vec / np.array(sqrt(src_vec[0]**2+src_vec[1]**2))
        dest_vec = dest_vec / np.array(sqrt(dest_vec[0]**2+dest_vec[1]**2))
        src_vec *= 100
        dest_vec *= 100

        src_vec = src_vec.astype(np.int32)
        dest_vec = dest_vec.astype(np.int32)
        joint_pos = joint_pos.astype(np.int32)

        points = np.array([joint_pos, joint_pos + src_vec, joint_pos + dest_vec], np.int32)
        points = points.reshape((-1, 1, 2))

        if angle_val < 0.3:
            color = (100, 100, 255)
        elif angle_val < 0.7:
            color = (120, 220, 235)
        else:
            color = (140, 215, 140)

        edge_color = np.array(color) * 0.7

        overlay = rgb_image.copy()


        cv2.polylines(overlay, [points], True, color)
        cv2.fillPoly(overlay, [points], color)

        alpha = 0.6

        result = cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0, rgb_image)

        cv2.circle(result, joint_pos, radius=round(4 * annotation_coefficient), color=edge_color, thickness=-1)
        cv2.circle(result, joint_pos + src_vec, radius=round(4 * annotation_coefficient), color=edge_color, thickness=-1)
        cv2.circle(result, joint_pos + dest_vec, radius=round(4 * annotation_coefficient), color=edge_color, thickness=-1)

        cv2.line(result, joint_pos, joint_pos + src_vec, thickness=round(3*annotation_coefficient), color=edge_color)
        cv2.line(result, joint_pos, joint_pos + dest_vec, thickness=round(3*annotation_coefficient), color=edge_color)

        return result



    elif 41 <= element_id <= 52:
        if not pose_data[frame_index]:
            return rgb_image

        keypoint_pos = pose_data[frame_index]["keypoints"][element_id - 36]
        keypoint_pos[0] = int(keypoint_pos[0])
        keypoint_pos[1] = int(keypoint_pos[1])
        xv, yv = analytics_data[element_id]["data"][frame_index][0]
        vel = [int(xv * 0.2), int(yv * 0.2)]

        result = cv2.line(rgb_image, keypoint_pos, (np.array(keypoint_pos) + np.array(vel)).tolist(), (0, 255, 255), thickness=round(8 * annotation_coefficient))
        result = cv2.circle(result, (np.array(keypoint_pos) + np.array(vel)).tolist(), radius=round(8 * annotation_coefficient), color=(0, 255, 255), thickness=-1)
        return result

    elif 53 <= element_id <= 56:

        hold_index = analytics_data[element_id]["data"][frame_index]

        if not analytics_data[element_id] or hold_index == -1:
            return rgb_image


        color_map = [(80, 156, 131), (93, 147, 214), (224, 62, 179), (235, 81, 53)]
        p1 = hold_data[hold_index]["bbox"][:2]
        p2 = hold_data[hold_index]["bbox"][2:]
        return cv2.rectangle(rgb_image, [int(a) for a in p1], [int(a) for a in p2], color=color_map[element_id - 53], thickness=round(8 * annotation_coefficient))
    else:
        return rgb_image

def start_stop_button():
    global isVideoPaused
    if isVideoPaused:
        # Start video
        dpg.set_item_label("start_btn", "| |")
        isVideoPaused = False
    else:
        # Pause video
        dpg.set_item_label("start_btn", ">")
        isVideoPaused = True

def set_fps(sender):
    global PLAYBACK_FPS
    PLAYBACK_FPS = dpg.get_value(sender)

def on_key_pressed(sender, app_data):
    global isVideoPaused
    # app_data will contain the key that was pressed
    if app_data == dpg.mvKey_Spacebar:
        start_stop_button()



with dpg.handler_registry():
    dpg.add_key_press_handler(callback=on_key_pressed)


object_colormap = [
                (37, 37, 38),  # 0  = no contact (dark gray)

                # 1–47 = objects 0–46
                (230, 25, 75),
                (60, 180, 75),
                (255, 225, 25),
                (0, 130, 200),
                (245, 130, 48),
                (145, 30, 180),
                (70, 240, 240),
                (240, 50, 230),
                (210, 245, 60),
                (250, 190, 190),

                (0, 128, 128),
                (230, 190, 255),
                (170, 110, 40),
                (255, 250, 200),
                (128, 0, 0),
                (170, 255, 195),
                (128, 128, 0),
                (255, 215, 180),
                (0, 0, 128),
                (128, 128, 128),

                (255, 99, 71),
                (154, 205, 50),
                (70, 130, 180),
                (218, 112, 214),
                (255, 165, 0),
                (0, 191, 255),
                (186, 85, 211),
                (46, 139, 87),
                (255, 105, 180),
                (244, 164, 96),

                (72, 61, 139),
                (60, 179, 113),
                (123, 104, 238),
                (32, 178, 170),
                (219, 112, 147),
                (176, 196, 222),
                (188, 143, 143),
                (135, 206, 235),
                (255, 182, 193),
                (95, 158, 160),

                (175, 238, 238),
                (152, 251, 152),
                (221, 160, 221),
                (255, 228, 181),
                (176, 224, 230),
                (240, 230, 140),
                (255, 160, 122),
            ]
with dpg.colormap_registry():
    dpg.add_colormap(
        tag="object_contacts",
        colors=object_colormap,
        qualitative=True
    )

with dpg.theme() as red_theme:
    with dpg.theme_component(dpg.mvProgressBar):
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (255, 100, 100))

with dpg.theme() as yellow_theme:
    with dpg.theme_component(dpg.mvProgressBar):
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (235, 220, 120))

with dpg.theme() as green_theme:
    with dpg.theme_component(dpg.mvProgressBar):
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (140, 215, 140))

#---------------------------------
#    Update video method
#---------------------------------

def update_frame():
    global do_visualize_keypoints
    global do_visualize_bones
    global current_move_index

    frame_idx = int(dpg.get_value("frame_data"))

    if isVideoPaused or dpg.is_item_active("frame_data"):
        frame = Video.get_frame(frame_idx)

    else: # Frame slider is inactive
        frame = Video.next()
        dpg.set_value("frame_data", Video.get_index())
        frame_idx = int(Video.get_index())


    current_move_index = summary_data["FrameMoveMap"][frame_idx]



    if do_visualize_holds:
        frame = visualize_holds(frame)

    if do_display_current_move:
        frame = display_move(current_move_index, frame)

    if do_visualize_bones:
        frame = visualize_bones(frame_idx, frame)

    if do_visualize_keypoints:
        frame = visualize_keypoints(frame_idx, frame)

    display_analytics_panel(frame_idx)

    # Start visualizing the elements
    for k in range(len(visualize_elements)):  # loop through elements that must be visualized
        if visualize_elements[k]:  # if it is true
            frame = visualize(k, frame, frame_idx)  # visualize that element (the index of the element is the id as well)

    if do_display_move_index:
        cv2.rectangle(frame, (10, 10), (220, 80), 0, -1)

        if current_move_index >= 0:
            cv2.putText(frame, str(current_move_index), (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 7, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{abs(current_move_index) - 2}-{abs(current_move_index) - 1}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (180, 180, 180), 7, cv2.LINE_AA)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (texture_width, texture_height))
    texture_data = frame.astype(np.float32).flatten() / 255.0
    # Update the texture
    dpg.set_value("video_texture", texture_data)

    return frame_idx

def jump_to_frame(sender, app_data):
    global isVideoPaused
    if not dpg.is_item_hovered("Wall Contacts Plot"):
        return

    x, y = dpg.get_plot_mouse_pos()

    col = int(x)  # frame index
    row = int(y)

    dpg.set_value("frame_data", col)
    isVideoPaused = False
    start_stop_button()

    dpg.set_value("v"+str(56 - row), True)
    visualize_element("v"+str(56 - row), True)

    dpg.set_value("tabs", "detailedTab")



def prev_move():
    global current_move_index
    global isVideoPaused

    if current_move_index < 0:
        current_move_index / abs(current_move_index) - 2

    new_move_idx = (current_move_index - 1) % len(summary_data["RouteSegmentation"])
    dpg.set_value("frame_data", summary_data["RouteSegmentation"][new_move_idx]["start_frame"])
    isVideoPaused = False
    start_stop_button()

def next_move():
    global current_move_index
    global isVideoPaused

    if current_move_index < 0:
        current_move_index / abs(current_move_index) - 2

    new_move_idx = (current_move_index + 1) % len(summary_data["RouteSegmentation"])
    dpg.set_value("frame_data", summary_data["RouteSegmentation"][new_move_idx]["start_frame"])
    isVideoPaused = False
    start_stop_button()




#---------------------------------------------------
#                 Main window
#---------------------------------------------------

with (dpg.window(tag="Primary Window")):

    dpg.add_tab_bar(tag="tabs")

    RT = dpg.add_tab(label="Detailed", parent="tabs", tag="detailedTab")
    SUM = dpg.add_tab(label="Summary", parent="tabs", tag="summaryTab")

    with dpg.group(horizontal=True, parent=RT):

        # ---------------------------------------------------------------
        #                        Visualizations
        # ---------------------------------------------------------------

        with dpg.child_window(width=400, border=False):

            with dpg.child_window(tag="visualizations", height=viewport_height - 230, menubar=True, resizable_y=True):

                with dpg.menu_bar():
                    dpg.add_menu(label="Visualizations", enabled=False)

                for i in range(len(analytics_data)):

                    element = analytics_data[i]

                    if not dpg.does_item_exist("v" + element["category"]): # if the category header doesn't exist, make it
                        with dpg.collapsing_header(label=element["category"], tag="v" + element["category"], default_open=True):
                            dpg.add_spacer(height=5)
                            dpg.add_checkbox(label="Display critical", tag="tb" + element["category"], indent=10)
                            dpg.add_spacer(height=15)
                        dpg.add_spacer(height=10)


                    dpg.add_spacer(height=5, parent="v" + element["category"])
                    dpg.add_checkbox(label=element["name"], tag="v" + str(i), callback=visualize_element,
                                     parent="v" + element["category"], indent=10, default_value=False)



            with dpg.child_window(tag="detected_data", menubar=True, height=130, resizable_y=True):

                with dpg.menu_bar():
                    dpg.add_menu(label="External Data", enabled=False)

                dpg.add_checkbox(label="Visualize keypoints", indent=10, callback=set_visualize_keypoints, default_value=do_visualize_keypoints)

                dpg.add_checkbox(label="Visualize bones", indent=10, callback=set_visualize_bones, default_value=do_visualize_bones)

                dpg.add_checkbox(label="Visualize holds", indent=10, callback=set_visualize_holds, default_value=do_visualize_holds)

            with dpg.child_window(tag="move_tab", menubar=True):
                with dpg.menu_bar():
                    dpg.add_menu(label="Move Tab", enabled=False)

                dpg.add_checkbox(label="Display Move Index", indent=10, callback=set_display_move_index, default_value=do_display_move_index)
                dpg.add_checkbox(label="Display Current Move", indent=10, callback=set_display_current_move, default_value=do_display_current_move)


        dpg.add_spacer(width=100)


        #-------------------------------------------------------------------
        #                           Video display
        # -------------------------------------------------------------------

        with dpg.child_window(width=texture_width, border=False):
            # Video display
            dpg.add_image("video_texture", width=texture_width, height=texture_height)

            # Frame Slider
            dpg.add_slider_int(tag="frame_data", min_value=1, max_value=Video.FRAME_COUNT, format="Frame Index: %d", no_input=True, width=texture_width)
            dpg.add_slider_int(tag="fps_slider", min_value=5, max_value=Video.FPS, format="Playback FPS: %d", no_input=True, callback=set_fps, default_value=Video.FPS, width=texture_width)

            with dpg.group(horizontal=True, indent=130):
                dpg.add_button(tag="prev_move", label="previous move", height=-1, callback=prev_move)
                dpg.add_button(tag="start_btn", label="| |", width=50, height=-1, callback=start_stop_button)
                dpg.add_button(tag="next_move", label="next move", height=-1, callback=next_move)


        dpg.add_spacer(width=100)

        #---------------------------------------------------------------
        #                        Analytics
        # ---------------------------------------------------------------

        with dpg.child_window(tag="analytics", menubar=True):

            with dpg.menu_bar():
                dpg.add_menu(label="Analytics", enabled=False)

            for i, element in enumerate(analytics_data[12:53]):

                j = i + 12

                if not dpg.does_item_exist(element["category"]): # if the category header doesn't exist, make it
                    dpg.add_collapsing_header(label=element["category"], tag=element["category"], default_open=True)
                    dpg.add_spacer(height=10)

                dpg.add_spacer(height=5, parent=element["category"])
                with dpg.group(horizontal=True, parent=element["category"]):
                    dpg.add_text(element["name"] + ": ")

                    if 52 >= j >= 12: # Com velocity, Joint angle and velocity, keypoint velocities
                        dpg.add_progress_bar(tag="e" + str(j))
                    else:
                        dpg.add_text("", tag="e" + str(j))

    with dpg.group(parent=SUM):

        with dpg.group(tag="HoldUsageGroup", horizontal=True):

            with dpg.plot(label ="Hold Usage Plot", tag="Hold Usage Plot", height=500, width=480, no_inputs=True):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvYAxis, no_label=True, no_gridlines=True, no_tick_labels=True)
                with dpg.plot_axis(dpg.mvXAxis, no_label=True, no_gridlines=True, no_tick_labels=True):
                    dpg.add_pie_series(0.5, 0.5, 0.4, [element for element in summary_data["HoldUsageSummary"].values()], ["0 limbs", "1 limbs", "2 limbs", "3 limb", "4 limbs", ], format="%.1f")

            with dpg.plot(label ="Wall Contacts Plot" ,tag="Wall Contacts Plot", height=500, width=-1):


                def encode(obj_id):
                    return 0 if obj_id is None else obj_id + 1

                raw_contact_data = []
                for limb_data in summary_data["HoldUsageMap"].values():
                    raw_contact_data += limb_data

                heatmap_data = [encode(obj_id) for obj_id in raw_contact_data]


                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Frame")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Limb")
                dpg.set_axis_limits_constraints(y_axis, 0, 4)
                dpg.set_axis_limits_constraints(x_axis, 0, Video.FRAME_COUNT - 1)
                dpg.set_axis_zoom_constraints(y_axis, 4, 4)

                dpg.set_axis_ticks(y_axis, (("Left Hand", 3.5), ("Right Hand", 2.5), ("Left Foot", 1.5), ("Right Foot", 0.5)))

                dpg.add_plot_legend(outside=True)
                dpg.add_heat_series(
                    x=heatmap_data,
                    rows=4,
                    cols=Video.FRAME_COUNT,
                    scale_min=0,
                    scale_max=47,
                    bounds_min=[0,0],
                    bounds_max=[Video.FRAME_COUNT - 1, 4],
                    parent=y_axis,
                    format=""
                )

                with dpg.handler_registry():
                    dpg.add_mouse_click_handler(callback=jump_to_frame)

                dpg.bind_colormap("Wall Contacts Plot", "object_contacts")

        with dpg.group(tag="CoMPlots", horizontal=True):
            pass

            #with dpg.plot(tag="")



dpg.bind_font(font)

#-------------------------------------------------
#            Setup and Main Loop
#-------------------------------------------------

dpg.create_viewport(title='Video', width=viewport_width, height=viewport_height)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

dpg.maximize_viewport()

dpg.render_dearpygui_frame()

while dpg.is_dearpygui_running():
    start_time = time.time()

    update_frame()

    dpg.render_dearpygui_frame()

    elapsed = time.time() - start_time
    sleep_time = 1 / PLAYBACK_FPS - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

dpg.destroy_context()
