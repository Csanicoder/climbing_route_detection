import time

import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from dearpygui.dearpygui import get_value

video_path = "out.mp4"
cap = cv2.VideoCapture(video_path)

target_fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / target_fps

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

# Read first frame to get size
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame from video.")

# Convert to RGB for DearPyGui
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#frame = cv2.rotate(frame, cv2.ROTATE_180)
width = 360
height = 640
frame = cv2.resize(frame, (width, height))

# Prepare normalized float32 texture data
texture_data = frame.astype(np.float32).flatten() / 255.0

# Create DearPyGui context
dpg.create_context()

with dpg.font_registry():
    font = dpg.add_font("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 24)

# Texture registry
with dpg.texture_registry():
    dpg.add_raw_texture(width, height, texture_data,
                        format=dpg.mvFormat_Float_rgb,
                        tag="video_texture")


#---------------------------------
#    Update video method
#---------------------------------

def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        # Restart video when finished
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return

    # Convert BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame = cv2.resize(frame, (width, height))

    # Ensure frame matches texture size
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))

    # Normalize to [0, 1] float32
    texture_data = frame.astype(np.float32).flatten() / 255.0

    # Update the texture
    dpg.set_value("video_texture", texture_data)


    return cap.get(cv2.CAP_PROP_POS_FRAMES) - 1


isVideoPaused = False

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


def set_frame_index(sender):
    global frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, get_value(sender))
    frame_idx = int(update_frame())


def check_frame_slider_active():
    return dpg.is_item_active("frame_data")


def update_data(item_tag, value):
    dpg.set_value(item_tag, value)

#---------------------------------------------------
#                 Main window
#---------------------------------------------------

texture_width = int(width * 1.5)
texture_height = int(height * 1.5)

with dpg.window(tag="Primary Window"):

    with dpg.group(horizontal=True):

        with dpg.child_window(width=texture_width + 20):
            # Video display
            dpg.add_image("video_texture", width=texture_width, height=texture_height)

            # Frame Slider
            dpg.add_slider_int(tag="frame_data", min_value=1, max_value=frame_count, format="Frame Index: %d", callback=set_frame_index, no_input=True, width=texture_width)

            # Start button
            dpg.add_button(tag="start_btn", label="| |", width=50, height=50, callback=start_stop_button)

        dpg.add_spacer(width=20)

        with dpg.group():
            #---------------------------------------------------------------
            #                        Analytics
            # ---------------------------------------------------------------
            with dpg.child_window(tag="analytics", no_scrollbar=True, height=600, menubar=True, resizable_y=True):
                with dpg.menu_bar():
                    dpg.add_menu(label="Analytics", enabled=False)

                with dpg.collapsing_header(label="Center of Mass", default_open=True):
                    dpg.add_text("Distance of CoM from Wall: 10cm")
                    dpg.add_text("Balance of CoM: Bottom Left")

                dpg.add_spacer(height=20)

                with dpg.collapsing_header(label="Angles", default_open=True):
                    dpg.add_text("Elbow Angle: 60 degrees")
                    dpg.add_text("Knee Angle: 60 degrees")

                dpg.add_spacer(height=20)

                with dpg.collapsing_header(label="Wall Contact", default_open=True):
                    dpg.add_text("Number of contact points: 4")
                    dpg.add_text("Average wall contact: 3.4")

                dpg.add_spacer(height=20)

                with dpg.collapsing_header(label="Hold Usage", default_open=True):
                    dpg.add_text("Jug usage: 4 sec")
                    dpg.add_text("Sloper usage: 12.6 sec")

            with dpg.child_window(tag="warnings", no_scrollbar=True, menubar=True):
                with dpg.menu_bar():
                    dpg.add_menu(label="Warnings", enabled=False)

                dpg.add_text("The CoM is overly positioned on the lower left region", color=(255, 0, 0, 255))
                dpg.add_text("Too much time spent in contact with slopers", color=(255, 155, 0, 255))
                dpg.add_text("Knee angle too large", color=(255, 240, 0, 255))

dpg.bind_font(font)

#-------------------------------------------------
#            Setup and Main Loop
#-------------------------------------------------

dpg.create_viewport(title='Video', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

dpg.maximize_viewport()

dpg.render_dearpygui_frame()

while dpg.is_dearpygui_running():
    start_time = time.time()

    if not check_frame_slider_active() and not isVideoPaused:
        frame_idx = int(update_frame())

    update_data("frame_data", frame_idx)

    dpg.render_dearpygui_frame()

    elapsed = time.time() - start_time
    sleep_time = frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

dpg.destroy_context()
cap.release()
