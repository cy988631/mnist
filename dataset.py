import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
train_dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transform
)
test_dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size = 128,
    shuffle = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size = 128,
    shuffle = False
)