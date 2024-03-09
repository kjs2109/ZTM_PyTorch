"""
Contains PyTorch model code to instantiate a TinyVGG model from the CNN Explainer website. 
"""
import torch 
from torch import nn 

class TinyVGG(nn.Module): 
    """Creates the TinyVGG architecture. 
    
    Replicates the TinyVGG architectoure from CNN explainer website in PyTorch. 
    See the original architecture here: https://poloclub.github.io/cnn-explainer/ 
    
    Args: 
        input_shape: 입력 채널 수 
        hidden_units: 중간 layer의 채널 수
        output_shape: 출력 벡터의 차원 수 (=target class의 수) 
    
    """
    def __init__(self, input_shape, hidden_unit, output_shape): 
        super().__init__() 
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_unit, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_unit, out_channels=hidden_unit, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) 
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit, out_channels=hidden_unit, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.Conv2d(in_channels=hidden_unit, out_channels=hidden_unit, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=hidden_unit*13*13, out_features=output_shape) 
        )
        
    def forward(self, x): 
        x = self.layer_1(x) 
        # print(x.shape)
        x = self.layer_2(x) 
        # print(x.shape)
        x = self.classifier(x) 
        return x
