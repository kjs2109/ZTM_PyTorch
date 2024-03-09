"""
Contains functionality for creating PyTorch DataLoader's for image classification data. 
"""
import os 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

NUM_WORKERS = 0  # os.cpu_count() 

def create_dataloaders(train_dir, test_dir, transform, batch_size): 
    """Create training and testing DataLoader 
    
    Takes in a training directory and testing directory path 
    and turns them into PyTorch Datasets and then into PyTorch DataLoaders 
    
    Args: 
        train_dir: train 데이터 폴더의 경로 
        test_dir: test 데이터 폴더의 경로 
        transform: 데이터에 적용될 transform 
        batch_size: 배치 사이즈 
        num_workers: DataLoader에서 사용할 cpu 개수 
        
    Return: 
        (train_dataloader, test_dataloader, class_names)가 튜플 형태로 반환 
        class_names는 target_class의 이름이 저장된 리스트  
    """
    
    train_data = datasets.ImageFolder(train_dir, transform) 
    test_data = datasets.ImageFolder(test_dir, transform) 
    
    class_names = train_data.classes 
    
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS
    )
    
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    return train_dataloader, test_dataloader, class_names
