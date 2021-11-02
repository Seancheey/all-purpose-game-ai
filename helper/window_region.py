import platform
from dataclasses import dataclass
from typing import Dict

import pywintypes
from screeninfo import screeninfo


@dataclass
class WindowRegion:
    x: int
    y: int
    width: int
    height: int

    @staticmethod
    def from_first_monitor():
        monitors = screeninfo.get_monitors()
        assert len(monitors) > 0, OSError('No Monitor Detected.')
        monitor = monitors[0]
        return WindowRegion(monitor.x, monitor.y, monitor.width, monitor.height)

    @staticmethod
    def from_window_with_name(window_title: str):
        if platform.system() != 'Windows':
            raise NotImplementedError("Systems other than Windows are not support for getting window region by name.")
        import win32gui
        import ctypes
        try:
            window = win32gui.FindWindow(None, window_title)
            desktop = win32gui.GetDC(window)
            gdi32 = ctypes.WinDLL("gdi32")
            visual_pixel = gdi32.GetDeviceCaps(desktop, 10)  # flag value for getting visual pixel height
            real_pixel = gdi32.GetDeviceCaps(desktop, 117)  # flag value for getting real pixel height
            dpi_scale = real_pixel / visual_pixel

            rect = win32gui.GetWindowRect(window)
            return WindowRegion(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]).scale_dpi(dpi_scale)
        except pywintypes.error:
            raise OSError(f"Unable to find the window with title \"{window_title}\"")

    def scale_size(self, ratio: float):
        new_width = round(self.width * ratio)
        new_height = round(self.height * ratio)
        new_x = self.x - (new_width - self.width) // 2
        new_y = self.y - (new_height - self.height) // 2
        new_region = WindowRegion(x=new_x, y=new_y, width=new_width, height=new_height)
        return new_region

    def scale_dpi(self, dpi_scale: float):
        return WindowRegion(
            x=int(self.x * dpi_scale),
            y=int(self.y * dpi_scale),
            width=int(self.width * dpi_scale),
            height=int(self.height * dpi_scale)
        )

    def to_mss_bounding_box(self) -> Dict[str, int]:
        return {
            'left': self.x,
            'top': self.y,
            'width': self.width,
            'height': self.height
        }
