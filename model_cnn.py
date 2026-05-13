import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.flattenn = nn.Flatten()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Conv2d(1,32,3,padding=1)
        self.fc2 = nn.Conv2d(32,64,3,padding=1)
        self.fc3 = nn.Linear(3136,10)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flattenn(x)
        x = self.fc3(x)
        return x