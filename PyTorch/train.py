""" 
Trains a PyTorch image classification model using device-agnostic code. 
"""
import os 
import torch 
from going_modular import data_setup, engine, model_builder, utils 

from torchvision import transforms 


torch.manual_seed(42) 
torch.cuda.manual_seed(42) 

# Setup hyperparameters 
save_model_name = "05_going_modular_script_mode_tinyvgg_model.pth"

NUM_EPOCHS = 5 
BATCH_SIZE = 32 
HIDDEN_UNITS = 10 
LEARNING_RATE = 0.001 

train_dir = "data/pizza_steak_sushi/train" 
test_dir = "data/pizza_steak_sushi/test" 

device = "cuda" if torch.cuda.is_available() else "cpu" 

data_transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor() 
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE) 

model = model_builder.TinyVGG(input_shape=3, hidden_unit=10, output_shape=len(class_names)).to(device) 

loss_fn = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

engine.train(model=model, 
             train_dataloader=train_dataloader, 
             test_dataloader=test_dataloader, 
             loss_fn=loss_fn, 
             optimizer=optimizer, 
             epochs=NUM_EPOCHS, 
             device=device) 

utils.save_model(model=model, target_dir="saved_models", model_name=save_model_name) 
