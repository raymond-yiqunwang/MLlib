import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, batch_size=64, train_ratio=0.7,
                              val_ratio=0.15, test_ratio=0.15, return_test=False,
                              num_workers=1, pin_memory=False):
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    train_split = int(np.floor(total_size * train_ratio))
    train_sampler = SubsetRandomSampler(indices[:train_split])
    val_split = train_split + int(np.floor(total_size * val_ratio))
    val_sampler = SubsetRandomSampler(
                    indices[train_split:val_split])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[val_split:])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    else:
        test_loader = None
    return train_loader, val_loader, test_loader


class XXNetDataset(Dataset):
    def __init__(self, root):
        self.root = root

    def __getitem__(self, idx):
        # load point_cloud
        data = pd.read_csv(self.root)
        return torch.Tensor(data.iloc[idx, :-1]), torch.Tensor([data.iloc[idx, -1]])

    def __len__(self):
        data = pd.read_csv(self.root)
        return data.shape[0]


