from dataclasses import dataclass
from datetime import datetime
from os import path

from torch.utils.tensorboard import SummaryWriter


@dataclass
class Summarizer:
    """
    Writes various summary information about model to tensor board log, so it could be read be tensor board.
    """
    train_log_dir: str
    train_name: str
    train_loss_record_interval = 1
    __train_loss_step = 0
    __test_loss_step = 0
    __train_loss_loop_i = 1
    __writer: SummaryWriter = None

    def __post_init__(self):
        self.__writer = SummaryWriter(
            path.join(self.train_log_dir, self.train_name + "-" + datetime.now().strftime('%Y%m%d-%H-%M-%S')))

    def add_graph(self, model, sample_data_input):
        self.__writer.add_graph(model, sample_data_input)

    def add_train_loss(self, loss):
        if self.__train_loss_loop_i < self.train_loss_record_interval:
            self.__train_loss_loop_i += 1
        else:
            self.__train_loss_loop_i = 1
            self.__writer.add_scalar('train loss', loss, self.__train_loss_step)
        self.__train_loss_step += 1

    def add_test_loss(self, loss):
        self.__writer.add_scalar('test loss', loss, self.__test_loss_step)
        self.__test_loss_step += 1
