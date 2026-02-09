import dearpygui.dearpygui as dpg
import json
import time

dpg.create_context()

with open("../data/blue_v6_analytics.json") as f:
    analytics_data = json.load(f)

points = analytics_data[22]["data"][100:1130]

# State variables
is_playing = False
current_index = 0

# Timing variables
target_fps = 30
frame_duration = 1.0 / target_fps  # ~0.0333 seconds
last_update_time = time.time()

def toggle_animation():
    global is_playing
    is_playing = not is_playing

def update_point(sender, app_data):
    # app_data is the integer index from the slider
    idx = app_data

    # Directly grab the coordinate without any math
    new_x = points[idx][0]
    new_y = 1920 - points[idx][1]

    # Update the tracker point
    dpg.set_value("tracker_point", [[new_x], [new_y]])

with dpg.window(label="Trajectory Map", width=1800, height=1080):
    # Method A: Inside a Plot (Good for navigation/GPS)
    with dpg.plot(label="Path Trajectory", height=1920/2, width=1080/2, equal_aspects=True):
        dpg.add_plot_axis(dpg.mvXAxis, label="East/West", tag="x_axis")
        with dpg.plot_axis(dpg.mvYAxis, label="North/South", tag="y_axis"):
            # This will connect them in the order of the list
            dpg.add_line_series([p[0] for p in points], [1920 - p[1] for p in points], label="Robot Path")

            # The "Snapped" point
            dpg.add_scatter_series([points[0][0]], [1920 - points[0][1]], tag="tracker_point")

            # Simple theme to make the point stand out
            with dpg.theme() as theme_point:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (0, 255, 0, 255))
            dpg.bind_item_theme("tracker_point", theme_point)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Play/Pause", callback=toggle_animation)
                dpg.add_button(label="Reset", callback=lambda: globals().update(current_index=0))

                # Speed control
                dpg.add_slider_int(label="Animation Speed", default_value=1, min_value=1, max_value=5, tag="speed_ctrl")

        dpg.set_axis_limits("x_axis", 0, 1080)
        dpg.set_axis_limits("y_axis", 0, 1920)

        dpg.reset_axis_zoom_constraints("x_axis")
        dpg.reset_axis_zoom_constraints("y_axis")


    # Integer slider ranging from 0 to the last index of your list
    dpg.add_slider_int(
        label="Step Through Path",
        min_value=0,
        max_value=len(points) - 1,
        callback=update_point,
        width=-1
    )


dpg.create_viewport(title='Trajectory Demo', width=1800, height=1080)
dpg.setup_dearpygui()
dpg.show_viewport()

# --- THE MAIN RENDER LOOP ---
while dpg.is_dearpygui_running():
    current_time = time.time()

    if is_playing and (current_time - last_update_time) >= frame_duration:
        # Increase index based on speed
        speed = dpg.get_value("speed_ctrl")
        current_index += speed

        # Loop back to start if we finish the path
        if current_index >= len(points):
            current_index = 0

        # Update the point position
        new_x = points[current_index][0]
        new_y = 1920 - points[current_index][1]
        dpg.set_value("tracker_point", [[new_x], [new_y]])

        # Reset the timer
        last_update_time = current_time

    dpg.render_dearpygui_frame()

dpg.destroy_context()