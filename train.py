import torch
from torch import nn
from torch import optim
from dataset import train_loader
from dataset import test_loader
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
accuracy = []

def evaluate(dataloader,model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _,predicted = torch.max(outputs,dim = 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct/total
    return acc


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X,y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*X.size(0)
    accuracy.append(evaluate(test_loader,model))
    epoch_loss = running_loss/len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch{epoch+1},Loss:{epoch_loss:.4f}')
        
torch.save(model.state_dict(), 'mnist_MLP_model.pth')

plt.figure()
plt.plot(train_losses,label = 'Train Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

plt.figure()
plt.plot(accuracy,label = 'Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Curve')
plt.legend()

plt.show()

