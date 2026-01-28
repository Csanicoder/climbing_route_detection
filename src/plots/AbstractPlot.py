from abc import ABC, abstractmethod
import dearpygui.dearpygui as dpg

class AbstractPlot(ABC):
    def __init__(self, tag, parent_group, width : int = 300, height : int = 300):
        with dpg.plot(tag = tag, width=width, height=height, parent=parent_group):
            dpg.add_plot_axis(dpg.mvXAxis)
            dpg.add_plot_axis(dpg.mvYAxis)

    @abstractmethod
    def add_series(self,):
        pass