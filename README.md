### Pytorch MNIST dataset usage

This repo creates a simple MNIST model. Heres an usage parts.

## Installation

```bash
pip install torch torchvision```

## Setting up

# 1-Clone the repo.

```bash
git clone https://Melon3141/pytorch-mnist && cd pytorch-mnist ```

# 2-Train the model.
```bash
python main.py
```

And thats all! Now you have an MNIST model. If you wanna test the model:

```bash
python test.py```

And if you wanna try with your image:
```python
from PIL import Image
import train
import torch

image = Image.open("path/of/your/image")
model = torch.load("model.pt", weights_only=False)

pred = model(train.tf(image)[0].reshape(1, 28, 28)).argmax().item()
print(pred)
```