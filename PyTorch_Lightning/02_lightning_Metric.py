import torch 
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch import nn, optim 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from torch.utils.data import random_split 
import pytorch_lightning as pl 
import torchmetrics 
from torchmetrics import Metric 


class MyAccuracy(Metric): 
    def __init__(self): 
        super().__init__() 
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum") 
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum") 
        
    def update(self, preds, target): 
        preds = torch.argmax(preds, dim=1) 
        assert preds.shape == target.shape 
        self.correct += torch.sum(preds == target) 
        self.total += target.numel()
        
    def compute(self): 
        return self.correct.float() / self.total.float() 


class NN(pl.LightningModule): 
    def __init__(self, input_size, num_classes):     
        super().__init__() 
        self.fc1 = nn.Linear(input_size, 50) 
        self.fc2 = nn.Linear(50, num_classes) 
        
        self.loss_fn = nn.CrossEntropyLoss() 
        
        self.my_accuracy = MyAccuracy() 
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes) 
        
    def forward(self, x): 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x 
    
    def training_step(self, batch, batch_idx): 
        loss, scores, y = self._common_step(batch, batch_idx) 
        accuracy = self.my_accuracy(scores, y) 
        f1_score = self.f1_score(scores, y) 
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score}, 
                      on_step=False, on_epoch=True, prog_bar=True) 
        return loss 
    
    def validation_step(self, batch, batch_idx): 
        loss, scores, y = self._common_step(batch, batch_idx) 
        self.log('val_loss', loss)
        return loss 
    
    def test_step(self, batch, batch_idx): 
        loss, scores, y = self._common_step(batch, batch_idx) 
        self.log('test_loss', loss)
        return loss 
    
    def _common_step(self, batch, batch_idx): 
        x, y = batch 
        x = x.reshape(x.size(0), -1) 
        scores = self.forward(x) 
        loss = self.loss_fn(scores, y) 
        return loss, scores, y
    
    def predict_step(self, batch, batch_idx): 
        x, y = batch 
        x = x.reshape(x.shape(0), -1) 
        scores = self.forward(x) 
        preds = torch.argmax(scores, dim=1) 
        return preds 
    
    def configure_optimizers(self): 
        return optim.Adam(self.parameters(), lr=0.001)

    
    
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

# trainer = pl.Trainer(acceelerator="gpu", devices=[0, 1], min_epochs=1, max_epochs=3, precision=16, num_nodes=1) 
trainer = pl.Trainer(min_epochs=1, max_epochs=3, precision='bf16-mixed') 
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader) 
trainer.test(model, test_loader)
        
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

