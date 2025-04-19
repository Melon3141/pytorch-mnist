from model import MODEL
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28))
])
train_data = MNIST(root="data",
                          train=True,
                          download=True,
                          transform=tf
                          )
test_data = MNIST(root="data",
                         train=False,
                         download=True,
                         transform=tf
                          )
train = DataLoader(train_data, batch_size=64, shuffle=True)
test = DataLoader(test_data, batch_size=64, shuffle=True)

def train_loop(model, loss, optimizer, train_data, epoch):
  model.train()
  for j in range(epoch):
    for i, (x, y) in enumerate(train_data):
      pred = model(x)
      loss_value = loss(pred, y)
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()
    print(f"Epoch {j+1} is finished")
  print("Training is finished")