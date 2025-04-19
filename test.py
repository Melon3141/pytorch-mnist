# This is models testing part.

import torch
import train

model = torch.load("model.pt", weights_only=False)
trues = 0

for i in range(len(train.test_data)):
    model.eval()
    data = train.test_data[i][0].reshape(1, 28, 28)
    pred = model(data)
    if pred.argmax().item() == train.test_data[i][1]:
        trues += 1
print("Test is finished. Models Accurate: ", trues / len(train.test_data))

   
