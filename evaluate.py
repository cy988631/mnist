from dataset import test_loader
from model import MLP 
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = MLP().to(device)
model.load_state_dict(torch.load('mnist_MLP_model.pth'))
model.eval()
torch.no_grad()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for X,y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _,predicted = torch.max(outputs,dim = 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        predicted_numpy = predicted.cpu().numpy()
        y_numpy = y.cpu().numpy()
        
    accuracy = correct/total
    
print(f'准确率: {100 * accuracy:.2f}%')