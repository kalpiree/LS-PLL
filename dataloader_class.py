import torch
from torch.utils.data import Dataset


class CIFARPseudoDataset(Dataset):
    def __init__(self, feats, target):
        self.x = feats
        self.y = target

    # Getting the data samples
    def __getitem__(self, idx):
        sample = [torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]).squeeze()]
        return sample

    def __len__(self):
        return len(self.x)
