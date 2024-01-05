import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(LeNet5Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = LeNet5Block(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = LeNet5Block(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = LeNet5Block(16, 120, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Example usage:
def LeNet5Model():
    return LeNet5(num_classes=10)  # Update num_classes according to the dataset



