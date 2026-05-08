import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
    transforms.Normalize()
])