import json
import time
import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from dearpygui.dearpygui import get_value

import VideoClass
from src.create_analytics import body_centers

video_name = "red_lache_to_hook"

Video = VideoClass.Video(f"../video/{video_name}_fixed.mp4")

# Load Holds JSON file with hold data
with open(f"../data/{video_name}_holds.json") as hold_f:
    hold_data = json.load(hold_f)

# Load Pose JSON file with pose data
with open(f"../data/{video_name}_pose_smoothed.json") as pose_f:
    pose_data = json.load(pose_f)

# Load Pose JSON file with analytics
with open(f"../data/{video_name}_analytics.json") as analytics_f:
    analytics_data = json.load(analytics_f)


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
isVideoPaused = False

def set_visualize_keypoints(sender):
    global do_visualize_keypoints
    do_visualize_keypoints = dpg.get_value(sender)

def set_visualize_bones(sender):
    global do_visualize_bones
    do_visualize_bones = dpg.get_value(sender)

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

def display_analytics_panel(frame_index):
    for i in range(len(analytics_data)):
        display = 0

        if i == 22: # Center of Mass Position
            x, y = analytics_data[i]["data"][frame_index]
            display = str(int(x)) + ", " + str(int(y))

        if i == 23 or 52 >= i >= 41: # Velocity of Center of Mass, keypoint velocities
            display = analytics_data[i]["data"][frame_index][1] # magnitude of velocity vector, divide by arbitrary max velocity
            if i == 23:
                display /= 500
            else:
                display /= 1200

        if i == 24:
            if not analytics_data[i]["data"][frame_index]:
                continue
            display = analytics_data[i]["data"][frame_index][1] / 500

        if 25 <= i <= 32:
            display = analytics_data[i]["data"][frame_index] / 180

        if 33 <= i <= 40:
            display = max(0, analytics_data[i]["data"][frame_index] / 2000 + 0.5)



        dpg.set_value("e" + str(i), display)

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

    elif element_id <= 15:
        v = analytics_data[element_id]["data"][frame_index][0]
        if not v:
            return rgb_image

        step = element_id % 4
        body_center = analytics_data[1]["data"][frame_index]
        color = (255, 60 * step, 200 + 12 * step)
        p2 = [int(a + b) for a, b in zip(body_center, v)]
        result = cv2.line(rgb_image, [int(a) for a in body_center], p2, color=color, thickness=round(6 * annotation_coefficient))

    elif element_id <= 21:
        line = analytics_data[element_id]["data"][frame_index][0]
        if not line:
            return rgb_image

        step = element_id % 4
        if element_id <= 19:
            color = (60 * step, 255, 180 + 15 * step)
        else:
            color = (100 * (step + 1), 100 * (step + 1), 240)
        result = cv2.line(rgb_image, [int(a) for a in line[0]], [int(a) for a in line[1]], color=color, thickness=round(6 * annotation_coefficient))

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

    elif element_id == 24:
        line = analytics_data[element_id]["data"][frame_index][0]
        if not line:
            return rgb_image
        result = cv2.line(rgb_image, [int(a) for a in line[0]], [int(a) for a in line[1]], color=(40, 100, 255), thickness=round(10 * annotation_coefficient))

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

def on_space_pressed(sender, app_data):
    # app_data will contain the key that was pressed
    if app_data == dpg.mvKey_Spacebar:
        start_stop_button()

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=on_space_pressed)

#---------------------------------
#    Update video method
#---------------------------------

def update_frame():
    global do_visualize_keypoints
    global do_visualize_bones

    frame_idx = dpg.get_value("frame_data")

    if isVideoPaused or dpg.is_item_active("frame_data"):
        frame = Video.get_frame(frame_idx)

    else: # Frame slider is inactive
        frame = Video.next()
        dpg.set_value("frame_data", Video.get_index())

    if do_visualize_bones:
        frame = visualize_bones(frame_idx, frame)

    if do_visualize_keypoints:
        frame = visualize_keypoints(frame_idx, frame)

    display_analytics_panel(frame_idx)

    # Start visualizing the elements
    for k in range(len(visualize_elements)):  # loop through elements that must be visualized
        if visualize_elements[k]:  # if it is true
            frame = visualize(k, frame, frame_idx)  # visualize that element (the index of the element is the id as well)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (texture_width, texture_height))
    texture_data = frame.astype(np.float32).flatten() / 255.0
    # Update the texture
    dpg.set_value("video_texture", texture_data)

    return frame_idx


#---------------------------------------------------
#                 Main window
#---------------------------------------------------

with dpg.window(tag="Primary Window"):

    dpg.add_tab_bar(tag="tabs")

    RT = dpg.add_tab(label="Detailed", parent="tabs")
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

                    if not dpg.does_item_exist("v" + element["category"]): # if the category header doesn't exist, make it
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
            dpg.add_slider_int(tag="frame_data", min_value=1, max_value=Video.FRAME_COUNT, format="Frame Index: %d", no_input=True, width=texture_width)
            dpg.add_slider_int(tag="fps_slider", min_value=5, max_value=Video.FPS, format="Playback FPS: %d", no_input=True, callback=set_fps, default_value=Video.FPS, width=texture_width)

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=(texture_width - 50) / 2)
                dpg.add_button(tag="start_btn", label="| |", width=50, height=-1, callback=start_stop_button)


        dpg.add_spacer(width=100)

        #---------------------------------------------------------------
        #                        Analytics
        # ---------------------------------------------------------------

        with dpg.child_window(tag="analytics", no_scrollbar=True, menubar=True):

            with dpg.menu_bar():
                dpg.add_menu(label="Analytics", enabled=False)

            for i in range(len(analytics_data)):

                element = analytics_data[i]

                if not dpg.does_item_exist(element["category"]): # if the category header doesn't exist, make it
                    dpg.add_collapsing_header(label=element["category"], tag=element["category"], default_open=True)
                    dpg.add_spacer(height=10)

                dpg.add_spacer(height=5, parent=element["category"])
                with dpg.group(horizontal=True, parent=element["category"]):
                    dpg.add_text(element["name"] + ": ")

                    if 51 >= i >= 23: # Com velocity, Joint angle and velocity, keypoint velocities
                        dpg.add_progress_bar(tag="e" + str(i))
                    else:
                        dpg.add_text("", tag="e" + str(i))



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
