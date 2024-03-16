import torch 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch import nn, optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torch.utils.data import random_split 

class NN(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super().__init__() 
        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes) 
        
    def forward(self, x): 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x 
    
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# Hyperparameters 
input_size = 784 
num_classes = 10 
learning_rate = 0.001 
batch_size = 64 
num_epochs = 3 

# Load Data 
entire_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True) 
train_ds, val_ds = random_split(entire_dataset, [50000, 10000]) 
test_ds = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) 

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False) 
test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False) 

# Initialize network 
model = NN(input_size=input_size, num_classes=num_classes).to(device) 

# loss and optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# Train Network 
print("Start training!")
for epoch in range(num_epochs): 
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch+1}/{num_epochs}")): 
        data = data.to(device) 
        targets = targets.to(device) 
        
        data = data.reshape(data.shape[0], -1)  # maintain batch and flatten 
        
        scores = model(data) 
        loss = criterion(scores, targets)
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
def check_accuracy(loader, model): 
    num_correct = 0 
    num_samples = 0 
    model.eval() 
    
    with torch.inference_mode(): 
        for x, y in loader: 
            
            x, y = x.to(device), y.to(device) 
            
            x = x.reshape(x.shape[0], -1) 
            
            scores = model(x) 
            _, predictions = scores.max(1) 
            
            num_correct += (predictions == y).sum() 
            
            num_samples += predictions.size(0) 
            
    model.train() 
    return num_correct / num_samples 


model.to(device) 
print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on validation set: {check_accuracy(val_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

