import torch
import torch.nn as nn

class FashionDCNN_ConfigA(nn.Module):
    """
    Config A: 2 Conv Blocks (Standard DCNN)
    - Block 1: Conv2d(32) -> ReLU -> MaxPool2d
    - Block 2: Conv2d(64) -> ReLU -> MaxPool2d
    - Fully Connected Layers
    """
    def __init__(self):
        super(FashionDCNN_ConfigA, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FashionDCNN_ConfigB(nn.Module):
    """
    Config B: 3 Conv Blocks (Deeper DCNN)
    - Block 1: Conv2d(32) -> ReLU -> MaxPool2d
    - Block 2: Conv2d(64) -> ReLU -> MaxPool2d
    - Block 3: Conv2d(128) -> ReLU -> MaxPool2d
    - Fully Connected Layers
    """
    def __init__(self):
        super(FashionDCNN_ConfigB, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # ปรับ Padding ป้องกันมิติหลุด
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x