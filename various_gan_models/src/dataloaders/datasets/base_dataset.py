import abc

import torch.utils.data as data


class BaseDataset(data.Dataset, metaclass=abc.ABCMeta):
    def __init__(self, max_dataset_size: int, dataset_length: int) -> None:
        self.max_dataset_size = int(max_dataset_size)
        self.dataset_length = dataset_length

    def __len__(self) -> int:
        return max(self.dataset_length, self.max_dataset_size)
