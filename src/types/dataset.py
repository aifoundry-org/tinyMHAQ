from abc import abstractmethod

from src.types.dataloader import Dataloader


class Dataset:
    def __init__(self, data_dir: str, batch_size: int, device: str):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device

        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_train_dataloader(self) -> Dataloader:
        pass

    @abstractmethod
    def get_val_dataloader(Self) -> Dataloader:
        pass

    @abstractmethod
    def get_test_dataloader(self) -> Dataloader:
        pass
