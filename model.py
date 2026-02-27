"""
model.py — Simple CNN for MNIST Classification

A lightweight convolutional neural network designed for the MNIST handwritten
digit dataset (28x28 grayscale images, 10 classes). This model serves as the
target for FGSM adversarial attack evaluation.

Architecture:
    Conv2d(1, 32, 3) → ReLU → Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
    → Dropout(0.25) → Flatten → Linear(9216, 128) → ReLU → Dropout(0.5)
    → Linear(128, 10)
"""

import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    A simple CNN classifier for MNIST digits.

    Input:  (batch, 1, 28, 28) — grayscale images normalized to [0, 1]
    Output: (batch, 10)        — raw logits for each digit class (0-9)
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        # Classification layers
        # After conv1(28→26), conv2(26→24), pool(24→12): 64 * 12 * 12 = 9216
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch, 10).
        """
        x = F.relu(self.conv1(x))       # (batch, 32, 26, 26)
        x = F.relu(self.conv2(x))       # (batch, 64, 24, 24)
        x = self.pool(x)                # (batch, 64, 12, 12)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)       # (batch, 9216)
        x = F.relu(self.fc1(x))         # (batch, 128)
        x = self.dropout2(x)
        x = self.fc2(x)                 # (batch, 10)
        return x
