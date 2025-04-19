import torch
import torchvision
import numpy as np
from torch import nn
from torchvision.datasets import MNIST

class MODEL(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
  def forward(self, x):
    return self.layers(x)
