# data format configs

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageFormat:
    width: int
    height: int
    channel: int

    def np_shape(self):
        return self.height, self.width, self.channel

    def tensor_shape(self):
        return self.channel, self.height, self.width

    def resolution_shape(self):
        return self.width, self.height

    def __len__(self):
        return 3

    def __getitem__(self, item):
        return self.np_shape()[item]


img_size = ImageFormat(width=192, height=108, channel=3)
recording_keys = ['w', 'a', 's', 'd']
np_keys_filename = 'keys.npy'
np_screens_filename = 'screens.npy'
avi_video_filename = 'video.avi'
