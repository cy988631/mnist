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

correct = 0
total = 0

with torch.no_grad():
    for X,y in test_loader:
        outputs = model(X)
        _,predicted = torch.max(outputs,dim = 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    accuracy = correct/total
    
print(f'准确率: {100 * accuracy:.2f}%')