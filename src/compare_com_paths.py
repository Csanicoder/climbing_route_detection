import json
import dearpygui.dearpygui as dpg


def update_com_path_plot(sender, app_data):
    # app_data from a range slider is a list: [start_val, end_val]
    start, end = int(app_data[0]), int(app_data[1])

    if sender == "data_range_1":
        sliced_points = points_1[start:end + 1]
    else:
        sliced_points = points_2[start:end + 1]

    # Extract x and y
    x = [p[0] for p in sliced_points]
    y = [1920 - p[1] for p in sliced_points]

    # Update the existing series using its tag
    # If the tag doesn't exist yet, dpg.configure_item will do nothing
    if sender == "data_range_1":
        dpg.set_value("comPathPlot1", [x, y])
    else:
        dpg.set_value("comPathPlot2", [x, y])

route_name_1 = "red_easy"
route_name_2 = "black_v4"

with open(f"../data/{route_name_1}_analytics.json") as analytics_f1:
    analytics_data_1 = json.load(analytics_f1)

with open(f"../data/{route_name_2}_analytics.json") as analytics_f2:
    analytics_data_2 = json.load(analytics_f2)

points_1 = analytics_data_1[22]["data"] #CoM position
max_idx_1 = len(points_1) - 1

points_2 = analytics_data_2[22]["data"] #CoM position
max_idx_2 = len(points_2) - 1


dpg.create_context()
dpg.create_viewport(title='Compare CoM Paths', width=int(1000/1.5), height=int(1600/1.5))

with dpg.window(tag="Primary Window"):
    with dpg.group():
        with dpg.plot(label="Path Trajectory", height=1920 / 2, width=1080 / 2, equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="East/West", tag="x_axis")
            with dpg.plot_axis(dpg.mvYAxis, label="North/South", tag="y_axis"):
                # This will connect them in the order of the list
                dpg.add_line_series([p[0] for p in points_1], [1920 - p[1] for p in points_1], label="CoM Path",
                                    tag="comPathPlot1")

                dpg.add_line_series([p[0] for p in points_2], [1920 - p[1] for p in points_2], label="CoM Path",
                                    tag="comPathPlot2")

                dpg.set_axis_limits("x_axis", 0, 1080)
                dpg.set_axis_limits("y_axis", 0, 1920)

                dpg.reset_axis_zoom_constraints("x_axis")
                dpg.reset_axis_zoom_constraints("y_axis")

                dpg.add_drag_intx(
                    tag="data_range_1",
                    label=route_name_1,
                    size=2,
                    default_value=[0, max_idx_1],
                    min_value=0,
                    max_value=max_idx_1,
                    callback=update_com_path_plot
                )
                dpg.add_drag_intx(
                    tag="data_range_2",
                    label=route_name_2,
                    size=2,
                    default_value=[0, max_idx_2],
                    min_value=0,
                    max_value=max_idx_2,
                    callback=update_com_path_plot
                )

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()