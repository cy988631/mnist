import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
    transforms.Normalize((0.1307,),(0.3081,))
])
train_dataset = dataset.MNIST(
    root = 
    train = True,
    download = True,
    
)