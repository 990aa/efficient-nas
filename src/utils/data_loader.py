import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

class DatasetManager:
    """Manages multiple datasets for comprehensive benchmarking."""
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
        # Dataset statistics
        self.dataset_info = {
            'cifar10': {'num_classes': 10, 'input_size': (3, 32, 32)},
            'cifar100': {'num_classes': 100, 'input_size': (3, 32, 32)},
            'imagenet16-120': {'num_classes': 120, 'input_size': (3, 16, 16)},
            'fashion-mnist': {'num_classes': 10, 'input_size': (1, 28, 28)},
            'medical-mnist': {'num_classes': 6, 'input_size': (1, 28, 28)}
        }
    
    def get_cifar10_loaders(self, batch_size: int = 96, 
                           cutout: bool = False) -> Tuple[DataLoader, DataLoader]:
        """CIFAR-10 loaders with search/evaluation transforms."""
        # Search phase transforms (light augmentation)
        search_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Final evaluation transforms (strong augmentation)
        eval_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        if cutout:
            eval_transform.transforms.append(Cutout(n_holes=1, length=16))
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, 
            download=True, transform=search_transform)
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False,
            download=True, transform=eval_transform)
        
        # Split train into train/val for search
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def get_cifar100_loaders(self, batch_size: int = 96,
                            cutout: bool = False) -> Tuple[DataLoader, DataLoader]:
        """CIFAR-100 loaders for transfer learning evaluation."""
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761))
        ])
        
        if cutout:
            transform.transforms.append(Cutout(n_holes=1, length=16))
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True,
            download=True, transform=transform)
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False,
            download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def get_imagenet16_120_loaders(self, batch_size: int = 96) -> Tuple[DataLoader, DataLoader]:
        """ImageNet16-120 loaders for computational efficiency testing."""
        # Note: This requires downloading ImageNet16-120 dataset first
        # We'll use a placeholder that resizes CIFAR-100 for demonstration
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                               (0.229, 0.224, 0.225))
        ])
        
        # Using CIFAR-100 as proxy for demonstration
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=True,
            download=True, transform=transform)
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_root, train=False,
            download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def get_fashion_mnist_loaders(self, batch_size: int = 96) -> Tuple[DataLoader, DataLoader]:
        """Fashion-MNIST loaders for domain-specific testing."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_root, train=True,
            download=True, transform=transform)
        
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_root, train=False,
            download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        return train_loader, test_loader
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.dataset_info.get(dataset_name, {})

class Cutout:
    """Random mask cutout augmentation."""
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img