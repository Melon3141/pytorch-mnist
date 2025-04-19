from model import MODEL
import train
import torch
from torch import nn

mnist_model = MODEL().to("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.01
momentum = 0.9
epoch = int(input("Train Epoch: "))
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mnist_model.parameters(), lr=lr, momentum=momentum)

train.train_loop(mnist_model, loss, optimizer, train.train, epoch)

torch.save(mnist_model, "model.pt")
