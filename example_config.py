import os

from helper.data_format import ImageFormat
from helper.model import PlayModel
from helper.project_config import ProjectConfig
from helper.window_region import WindowRegion

linea_config = ProjectConfig(
    recording_keys=['w', 'a', 's', 'd'],
    img_format=ImageFormat(width=192, height=108, channel=3),
    data_dir=os.path.join(os.getcwd(), 'linea', 'data'),
    train_log_dir=os.path.join(os.getcwd(), 'linea', 'runs'),
    train_name='multi-class-train',
    record_window_region_func=WindowRegion.from_first_monitor,
    model_path=os.path.join(os.getcwd(), 'linea', 'model.pth'),
    model_class=lambda: PlayModel(num_outputs=4),
    max_record_fps=60,
    stop_train_key='ctrl+q',
    save_record_key='space',
    start_apply_key='e',
    stop_apply_key='q',
    start_record_key='e',
    stop_record_key='q',
    data_visualize_fps=20,
)

super_hexagon_config = ProjectConfig(
    recording_keys=['a', 'd'],
    img_format=ImageFormat(width=192, height=108, channel=3),
    data_dir=os.path.join(os.getcwd(), 'super_hexagon', 'data'),
    train_log_dir=os.path.join(os.getcwd(), 'super_hexagon', 'runs'),
    train_name='multi-class-train',
    record_window_region_func=lambda: WindowRegion.from_window_with_name('Super Hexagon').scale(0.6),
    model_path=os.path.join(os.getcwd(), 'super_hexagon', 'model.pth'),
    model_class=lambda: PlayModel(num_outputs=2),
    max_record_fps=60,
    stop_train_key='ctrl+q',
    save_record_key='space',
    start_apply_key='e',
    stop_apply_key='q',
    start_record_key='e',
    stop_record_key='q',
    data_visualize_fps=20,
)
