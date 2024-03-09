"""
Contains functions fo training and testing a PyTorch model 
"""
import torch 

def train_step(model, dataloader, loss_fn, optimizer, device): 
    """Trains a PyTorch model for a single epoch. 
    
    Turns a target PyTorch model to training mode and then runs through all of the required training steps 
    (forward pass, loss calculation, optimizer step) 
    
    Args: 
        model: A PyTorch model to be trained. 
        dataloader: A DataLoader instance for the model to be trained on. 
        loss_fn: A PyTorch loss function to minimize. 
        optimizer: A PyTorch optimizer to help minimize the loss function. 
        device: A target device to compute on 
    
    Returns: 
        A tuple of training loss and training accuracy metrics. 
        In the form (train_loss, train_accuracy)  
    """
    model.train() 
    train_loss, train_acc = 0, 0 
    for batch, (X, y) in enumerate(dataloader): 
        X, y = X.to(device), y.to(device) 
        
        y_pred = model(X) 
        
        loss = loss_fn(y_pred, y) 
        train_loss += loss.item() 
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=-1), dim=-1) 
        train_acc += (y_pred_class == y).sum().item() / len(X) 
        
    train_loss = train_loss / len(dataloader) 
    train_acc = train_acc / len(dataloader) 
    
    return train_loss, train_acc 


def test_step(model, dataloader, loss_fn, device): 
    """Tests a PyTorch model for a single epoch. 
    
    Turns a target PyTorch model to "eval" mode and then performs a forward pass on a testing dataset.  
    
    Args: 
        model: A PyTorch model to be tested. 
        dataloader: A DataLoader instance for the model to be tested on. 
        loss_fn: A PyTorch loss function to calculate loss on the test data. 
        device: A target device to compute on 
    
    Returns: 
        A tuple of testing loss and testing accuracy metrics. 
        In the form (test_loss, test_accuracy)    
    """
    model.eval() 
    test_loss, test_acc = 0, 0 
    with torch.inference_mode(): 
        for batch, (X, y) in enumerate(dataloader): 
            X, y = X.to(device), y.to(device) 
            
            test_pred_logits = model(X) 
            
            loss = loss_fn(test_pred_logits, y) 
            test_loss += loss.item() 
            
            test_pred_labels = test_pred_logits.argmax(dim=-1) 
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels)) 
            
        test_loss = test_loss / len(dataloader) 
        test_acc = test_acc / len(dataloader) 
        
    return test_loss, test_acc 


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device): 
    """Trains and tests a PyTorch model.  
    
    Passes a target PyTorch models through train_step() and test_step() functions for a number of epochs, 
    training and testing the model in the same epoch loop. 
    
    Callculates, prints and stores evaluation metrics throughout. 
    
    Args: 
        model: A PyTorch model to be trained and tested. 
        train_dataloader: A DataLoader instance for the model to be trained on. 
        test_dataloader: A DataLoader instance for the model to be tested on. 
        optimizer: A PyTorch optimizer to help minimize the loss function. 
        loss_fn: A PyTorch loss function to calculate loss on both datasets. 
        epochs: An integer indicating how many epochs to train for. 
        device: A target device to compute on 
        
    Returns: 
        A dictionary of training and testing loss as well as training and testing accuracy metrics. 
        Each metric has a value in a list for each epoch. 
    """
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} 
    
    for epoch in range(epochs): 
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device) 
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device) 
        
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | " 
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}") 
        
        results["train_loss"].append(train_loss) 
        results["train_acc"].append(train_acc) 
        results["test_loss"].append(test_loss) 
        results["test_acc"].append(test_acc)  
        
    return results 
    
