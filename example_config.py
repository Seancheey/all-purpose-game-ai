import os

from torchvision import transforms

from components.project_config import ProjectConfig
from components.utils.image_format import ImageFormat
from components.utils.window_region import WindowRegion
from model.hexagon_model import SuperHexagonModel
from model.linea_model import LineaModel


def linea_config(level):
    folder = 'linea-' + level
    return ProjectConfig(
        recording_keys=['w', 'a', 's', 'd'],
        img_format=ImageFormat(width=192, height=108, channel=3),
        data_dir=os.path.join(folder, 'data'),
        train_log_dir=os.path.join(folder, 'runs'),
        train_name='batch-norm-train',
        record_window_region_func=WindowRegion.from_first_monitor,
        model_path=os.path.join(folder, 'model.pth'),
        model_class=LineaModel,
        max_record_fps=60,
        stop_train_key='ctrl+q',
        save_record_key='space',
        start_apply_key='e',
        stop_apply_key='q',
        start_record_key='e',
        stop_record_key='q',
        data_visualize_fps=20,
        device='cuda'
    )


super_hexagon_config = ProjectConfig(
    recording_keys=['a', 'd'],
    img_format=ImageFormat(width=256, height=256, channel=1),
    data_dir=os.path.join(os.getcwd(), 'super_hexagon', 'data'),
    train_log_dir=os.path.join(os.getcwd(), 'super_hexagon', 'runs'),
    train_name='cnn-batch-norm-train',
    record_window_region_func=lambda: WindowRegion.from_window_with_name('Super Hexagon').scale_size(0.85),
    model_path=os.path.join(os.getcwd(), 'super_hexagon', 'model.pth'),
    model_class=SuperHexagonModel,
    max_record_fps=60,
    stop_train_key='ctrl+q',
    save_record_key='space',
    start_apply_key='e',
    stop_apply_key='esc',
    start_record_key='e',
    stop_record_key='esc',
    data_visualize_fps=30,
    screen_to_tensor_func=transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
    ]),
    screen_augmentation_func=transforms.RandomRotation(20),
    device='cuda'
)
