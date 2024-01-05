import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CIFARDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].astype(np.float32)  # Convert to float32
        label = self.labels[idx]

        # Reshape the feature to [3, 32, 32]
        feature = feature.reshape(3, 32, 32)

        # Convert to torch.Tensor
        feature = torch.from_numpy(feature).to(torch.float32)
        label = torch.tensor(label)

        # Apply the transform if it is provided
        if self.transform:
            feature = self.transform(feature)

        return feature, label



