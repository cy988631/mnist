import torch
from torch import nn
from torch import optim
from dataset import train_loader
from model import MLP

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epoches = 100
for epoch in range(num_epoches):
    for X,y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        print(f'Epoch{epoch+1},Loss:{loss.item():.4f}')
        
torch.save(model.state_dict(), 'mnist_MLP_model.pth')