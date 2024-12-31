
from typing import Tuple
from torch.utils.data import Subset


def split_dataset(dataset, train_ratio=0.8) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and test datasets
    
    Args:
        dataset: Dataset to split
        train_ratio: Train dataset ratio, default 0.8
        seed: Not used, keep interface consistent
        
    Returns:
        train_dataset, test_dataset
    """
    # Calculate train and test dataset sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    
    # Split indices in order
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    # Create subsets using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset