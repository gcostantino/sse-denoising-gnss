import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return x, y

    def __len__(self):
        count = self.x.shape[0]
        return count
