import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from numpy.random import randint

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=10, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(2356, 1000)
        self.fc2 = nn.Linear(1000, action_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
