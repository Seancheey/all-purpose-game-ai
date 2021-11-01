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
        handle = win32gui.FindWindow(None, window_title)
        rect = win32gui.GetWindowRect(handle)
        return WindowRegion(x=rect[0], y=rect[1], width=rect[2] - rect[0], height=rect[3] - rect[1])

    def to_mss_bounding_box(self) -> Dict[str, int]:
        return {
            'left': self.x,
            'top': self.y,
            'width': self.width,
            'height': self.height
        }
