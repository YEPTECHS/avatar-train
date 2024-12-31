import os
import wandb
import random
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn


def setup_wandb():
    wandb.login(key=os.getenv("WANDB_KEY"))
    wandb.init(project="musetalk-finetune")


def choose_random_outside_range(array, idx, exclusion_range=25):
    n = len(array)
    # Determine the range to exclude
    start = max(0, idx - exclusion_range)
    end = min(n, idx + exclusion_range + 1)
    
    # Build valid indices list
    valid_indices = list(range(0, start)) + list(range(end, n))
    
    if not valid_indices:
        raise ValueError("No valid indices outside the exclusion range!")
    
    # Randomly choose one from valid indices
    chosen_idx = random.choice(valid_indices)
    return chosen_idx


def frozen_params(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    
    # Verify model parameters are correctly frozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert trainable_params == 0, "Model parameters are not frozen!"    
    

def preprocess_img_tensor(image_tensor: torch.Tensor):
    # Assume input is a PyTorch tensor of shape (N, C, H, W)
    N, C, H, W = image_tensor.shape
    # Calculate new width and height, making them multiples of 32
    new_w = W - W % 32
    new_h = H - H % 32
    # Use torchvision.transforms library methods for scaling and resampling
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # Apply transformation to each image and store the result in a new tensor
    preprocessed_images = torch.empty((N, C, new_h, new_w), dtype=torch.float32)
    for i in range(N):
        # Use F.interpolate instead of transforms.Resize
        resized_image: torch.Tensor = F.interpolate(image_tensor[i].unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        preprocessed_images[i] = transform(resized_image.squeeze(0))

    return preprocessed_images
