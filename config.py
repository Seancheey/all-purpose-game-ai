import os

from helper.data_format import ImageFormat
from helper.project_config import ProjectConfig
from helper.window_region import WindowRegion

config = ProjectConfig(
    recording_keys=['w', 'a', 's', 'd'],
    img_format=ImageFormat(width=192, height=108, channel=3),
    data_dir=os.path.join(os.getcwd(), 'data'),
    train_name='multi-class-train',
    record_window_region=WindowRegion.from_first_monitor(),
    model_path=os.path.join(os.getcwd(), 'model.pth'),
    max_record_fps=30,
    stop_train_key='ctrl+q',
    save_record_key='space',
    start_apply_key='e',
    stop_apply_key='q',
    start_record_key='e',
    stop_record_key='q',
    data_visualize_fps=20,
)
