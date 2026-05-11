from dataset import test_loader
from model import MLP 
import torch

model = MLP()
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X,y in test_loader：
        outputs = model(X)
