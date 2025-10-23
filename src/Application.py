import json
import time

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from dearpygui.dearpygui import get_value, does_item_exist


Video = "blue_v6"

# Load Holds JSON file with hold data
with open(f"../data/{Video}_holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file with pose data
with open(f"../data/{Video}_pose_smoothed.json") as pose_f:
    pose_data = json.load(pose_f)

# Load Pose JSON file with analytics
with open(f"../data/{Video}_analytics.json") as analytics_f:
    analytics_data = json.load(analytics_f)


cap = 0
frame = 0
video_infos = {}

def init_video(video_path : str):
    global cap
    global frame
    global video_infos

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Get video infos
    video_infos = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS)
    }

    # Read first frame to get size
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")
init_video(f"../video/{Video}_fixed.mp4")

frame_time = 1 / video_infos["fps"]

viewport_width = 1920
viewport_height = 1080

texture_height = viewport_height - 100
texture_width = int((video_infos["width"] / video_infos["height"]) * texture_height)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (texture_width, texture_height))

texture_data = frame.astype(np.float32).flatten() / 255.0

# Create DearPyGui context
dpg.create_context()

with dpg.font_registry():
    font = dpg.add_font("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 18)

# Texture registry
with dpg.texture_registry():
    dpg.add_raw_texture(texture_width, texture_height, texture_data,
                        format=dpg.mvFormat_Float_rgb,
                        tag="video_texture")


visualize_elements = [False] * len(analytics_data)
refresh_visualize : bool = True
do_visualize_keypoints : bool = False
do_visualize_bones : bool = False
isVideoPaused = False

def set_visualize_keypoints(sender):
    global do_visualize_keypoints
    do_visualize_keypoints = get_value(sender)

def set_visualize_bones(sender):
    global do_visualize_bones
    do_visualize_bones = get_value(sender)

def visualize_keypoints(frame_index, rgb_image):
    global pose_data

    if not pose_data[frame_index]:
        return rgb_image

    result = rgb_image

    for keypoint in pose_data[frame_index]["keypoints"]:
        if pose_data[frame_index]["keypoints"].index(keypoint) in [0, 1, 2, 3, 4]:
            continue
        result = cv2.circle(rgb_image, [int(a) for a in keypoint], radius=8, color=(255, 255, 0), thickness=-1)

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
        result = cv2.line(rgb_image, keypoints[bone[0]], keypoints[bone[1]], (0, 255, 255), thickness=6)

    return result

def display_analytics_panel(frame_index):
    for i in range(len(analytics_data)):
        display = ""

        if i == 0: # Center of Mass Position
            x, y = analytics_data[i]["data"][frame_index]
            display = str(int(x)) + ", " + str(int(y))

        if i == 1 or 29 >= i >= 18: # Velocity of Center of Mass, keypoint velocities
            display = analytics_data[i]["data"][frame_index][1] # magnitude of velocity vector, divide by arbitrary max velocity
            if i == 1:
                display /= 500
            else:
                display /= 1200

        if 2 <= i <= 9:
            display = analytics_data[i]["data"][frame_index] / 180

        if 10 <= i <= 17:
            display = max(0, analytics_data[i]["data"][frame_index] / 2000 + 0.5)



        dpg.set_value("e" + str(i), display)

def visualize_element(sender, app_data): # callback for setting the display of elements
    visualize_elements[int(sender[1:])] = app_data

def visualize(element_id, rgb_image, frame_index):

    # Position of Center of Mass
    if element_id == 0:
        com_pos = analytics_data[0]["data"][frame_index]
        com_pos[0] = int(com_pos[0])
        com_pos[1] = int(com_pos[1])
        result = cv2.circle(rgb_image, com_pos, radius=12, color=(255, 0, 0), thickness=-1)
        return result


    # Velocity of Center of Mass
    if element_id == 1:
        com_pos = analytics_data[0]["data"][frame_index]
        com_pos[0] = int(com_pos[0])
        com_pos[1] = int(com_pos[1])
        xv, yv  = analytics_data[1]["data"][frame_index][0] # the velocity vector
        com_vel = [int(xv * 0.2), int(yv * 0.2)]
        result = cv2.line(rgb_image, com_pos, (np.array(com_pos) + np.array(com_vel)).tolist(), thickness=8, color=(0, 255, 0))
        result = cv2.circle(result, (np.array(com_pos) + np.array(com_vel)).tolist(), radius=8, color=(0, 255, 0), thickness=-1)
        return result

    if 18 <= element_id <= 29:
        if not pose_data[frame_index]:
            return rgb_image

        keypoint_pos = pose_data[frame_index]["keypoints"][element_id - 13]
        keypoint_pos[0] = int(keypoint_pos[0])
        keypoint_pos[1] = int(keypoint_pos[1])
        xv, yv = analytics_data[element_id]["data"][frame_index][0]
        vel = [int(xv * 0.2), int(yv * 0.2)]

        result = cv2.line(rgb_image, keypoint_pos, (np.array(keypoint_pos) + np.array(vel)).tolist(), (255, 255, 0), thickness=8)
        result = cv2.circle(result, (np.array(keypoint_pos) + np.array(vel)).tolist(), radius=8, color=(255, 255, 0), thickness=-1)
        return result


    return rgb_image

def start_stop_button(sender):
    global isVideoPaused
    if isVideoPaused:
        # Start video
        dpg.set_item_label(sender, "| |")
        isVideoPaused = False
    else:
        # Pause video
        dpg.set_item_label(sender, ">")
        isVideoPaused = True

#---------------------------------
#    Update video method
#---------------------------------

def update_frame():
    global cap
    global do_visualize_keypoints
    global do_visualize_bones

    if dpg.is_item_active("frame_data"): # If the frame slider is held down, we set the index of the next frame to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, dpg.get_value("frame_data"))
        refresh_visualize = True  # new frame, we must refresh the visualizations

    elif isVideoPaused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)


    ret, frame = cap.read()
    if not ret:
        # Restart video when finished
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = frame_rgb

    display_analytics_panel(frame_idx)

    if do_visualize_bones:
        result = visualize_bones(frame_idx, result)

    if do_visualize_keypoints:
        result = visualize_keypoints(frame_idx, result)

    # Start visualizing the elements
    for k in range(len(visualize_elements)):  # loop through elements that must be visualized
        if visualize_elements[k]:  # if it is true
            result = visualize(k, result, frame_idx)  # visualize that element (the index of the element is the id as well)




    result = cv2.resize(result, (texture_width, texture_height))

    # Normalize to [0, 1] float32
    texture_data = result.astype(np.float32).flatten() / 255.0

    # Update the texture
    dpg.set_value("video_texture", texture_data)


    return frame_idx


#---------------------------------------------------
#                 Main window
#---------------------------------------------------

with dpg.window(tag="Primary Window"):

    dpg.add_tab_bar(tag="tabs")

    RT = dpg.add_tab(label="Real-Time", parent="tabs")
    SUM = dpg.add_tab(label="Summary", parent="tabs")

    with dpg.group(horizontal=True, parent=RT):

        # ---------------------------------------------------------------
        #                        Visualizations
        # ---------------------------------------------------------------

        with dpg.child_window(width=400, border=False):

            with dpg.child_window(tag="visualizations", no_scrollbar=True, height=viewport_height - 130, menubar=True, resizable_y=True):

                with dpg.menu_bar():
                    dpg.add_menu(label="Visualizations", enabled=False)

                for i in range(len(analytics_data)):

                    element = analytics_data[i]

                    if not does_item_exist("v" + element["category"]): # if the category header doesn't exist, make it
                        dpg.add_collapsing_header(label=element["category"], tag="v" + element["category"], default_open=True)
                        dpg.add_spacer(height=10)

                    dpg.add_spacer(height=5, parent="v" + element["category"])
                    dpg.add_checkbox(label=element["name"], tag="v" + str(i), callback=visualize_element,
                                     parent="v" + element["category"], indent=10, default_value=False)



            with dpg.child_window(tag="detected_data", no_scrollbar=True, menubar=True):

                with dpg.menu_bar():
                    dpg.add_menu(label="External Data", enabled=False)

                dpg.add_checkbox(label="Visualize keypoints", indent=10, callback=set_visualize_keypoints, default_value=do_visualize_keypoints)

                dpg.add_checkbox(label="Visualize bones", indent=10, callback=set_visualize_bones, default_value=do_visualize_bones)


        dpg.add_spacer(width=100)


        #-------------------------------------------------------------------
        #                           Video display
        # -------------------------------------------------------------------

        with dpg.child_window(width=texture_width, border=False):
            # Video display
            dpg.add_image("video_texture", width=texture_width, height=texture_height)

            # Frame Slider
            dpg.add_slider_int(tag="frame_data", min_value=1, max_value=video_infos["frame_count"], format="Frame Index: %d", no_input=True, width=texture_width)

            # Start button
            dpg.add_button(tag="start_btn", label="| |", width=50, height=50, callback=start_stop_button)


        dpg.add_spacer(width=100)

        #---------------------------------------------------------------
        #                        Analytics
        # ---------------------------------------------------------------

        with dpg.child_window(tag="analytics", no_scrollbar=True, menubar=True):

            with dpg.menu_bar():
                dpg.add_menu(label="Analytics", enabled=False)

            for i in range(len(analytics_data)):

                element = analytics_data[i]

                if not does_item_exist(element["category"]): # if the category header doesn't exist, make it
                    dpg.add_collapsing_header(label=element["category"], tag=element["category"], default_open=True)
                    dpg.add_spacer(height=10)

                dpg.add_spacer(height=5, parent=element["category"])
                with dpg.group(horizontal=True, parent=element["category"]):
                    dpg.add_text(element["name"] + ": ")

                    if i == 0: # Center of Mass position
                        dpg.add_text("", tag="e" + str(i))
                    if 29 >= i >= 1: # Com velocity, Joint angle and velocity, keypoint velocities
                        dpg.add_progress_bar(tag="e" + str(i))



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

    frame_idx = int(update_frame())

    dpg.set_value("frame_data", frame_idx)

    dpg.render_dearpygui_frame()

    elapsed = time.time() - start_time
    sleep_time = frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

dpg.destroy_context()
cap.release()
