from dataset import test_loader
from model_mlp import MLP 
import torch
from sklearn.metrics import confusion_matrix as con
import matplotlib.pyplot as plt

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
        all_preds.extend(predicted_numpy)
        all_labels.extend(y_numpy)
        
    result = con(all_labels,all_preds)
    accuracy = correct/total
    
print(f'准确率: {100 * accuracy:.2f}%')

plt.imshow(result)
plt.colorbar()
plt.show()