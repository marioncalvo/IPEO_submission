import os
import torch
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader


def compute_normalization_stats(dataset_root, split="train", batch_size=16, num_workers=2):
    """
    Compute mean and standard deviation for dataset normalization
    
    Args:
        dataset_root (str): Path to the root directory containing train/validation/test folders
        split (str): Which split to use for computing stats, default: "train"
        batch_size (int): Batch size for DataLoader, default: 16
        num_workers (int): Number of workers for DataLoader, default: 2
    
    Returns:
        tuple: (mean, std) as torch tensors of shape (3,) for RGB channels
    """
    
    #Transform to resize images (all images need to have the same size) and convert to tensor 
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    
    #Load the specified split
    dataset_path = os.path.join(dataset_root, split)
    ds = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    #Compute mean
    mean_accumulator = torch.zeros(3)
    total_pixels = 0
    
    for batch_idx, (images, _) in enumerate(loader):
        images_reshaped = images.permute(0, 2, 3, 1).reshape(-1, 3)
        mean_accumulator += images_reshaped.sum(dim=0)
        total_pixels += images_reshaped.shape[0]
    
    mean = mean_accumulator / total_pixels
    
    #Compute standard deviation
    std_accumulator = torch.zeros(3)
    total_pixels = 0
    
    for batch_idx, (images, _) in enumerate(loader):
        images_reshaped = images.permute(0, 2, 3, 1).reshape(-1, 3)
        diff_squared = (images_reshaped - mean.unsqueeze(0)) ** 2
        std_accumulator += diff_squared.sum(dim=0)
        total_pixels += images_reshaped.shape[0]
    
    std = torch.sqrt(std_accumulator / total_pixels)
    
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std:  {std.tolist()}")
    
    return mean, std
