import platform
from dataclasses import dataclass
from typing import Dict

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
        import pywintypes
        handle = win32gui.FindWindow(None, window_title)
        try:
            rect = win32gui.GetWindowRect(handle)
        except pywintypes.error:
            rect = None
        if rect is None:
            raise OSError(f"Unable to find the window with title \"{window_title}\"")
        return WindowRegion(x=rect[0], y=rect[1], width=rect[2] - rect[0], height=rect[3] - rect[1])

    def scale(self, ratio: float):
        new_width = round(self.width * ratio)
        new_height = round(self.height * ratio)
        new_x = self.x - new_width // 2
        new_y = self.y - new_height // 2
        return WindowRegion(x=new_x, y=new_y, width=new_width, height=new_height)

    def to_mss_bounding_box(self) -> Dict[str, int]:
        return {
            'left': self.x,
            'top': self.y,
            'width': self.width,
            'height': self.height
        }
