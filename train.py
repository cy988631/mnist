import torch
from torch import nn
from torch import optim
from dataset import train_loader
from model import MLP
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_losses = []

def evaluate(dataloader,model):
    for X,y in dataloader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)


num_epoches = 100
for epoch in range(num_epoches):
    for X,y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print(f'Epoch{epoch+1},Loss:{loss.item():.4f}')
        
torch.save(model.state_dict(), 'mnist_MLP_model.pth')

plt.plot(train_losses,label = 'Train Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.legend
plt.savefig('loss_curve.png')

plt.show()

