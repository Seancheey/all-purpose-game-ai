# data format configs

img_size = (192, 108)
recording_keys = ['w', 'a', 's', 'd']
np_keys_filename = 'keys.npy'
np_screens_filename = 'screens.npy'
avi_video_filename = 'video.avi'

# below are helper variables/functions

key_to_index = {key: ind for key, ind in enumerate(recording_keys)}
index_to_key = {ind: key for key, ind in enumerate(recording_keys)}


def to_key_array(key_list):
    return [key in key_list for key in recording_keys]
