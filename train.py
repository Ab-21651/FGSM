"""
train.py — Train the MNIST CNN and Save Weights

Downloads the MNIST dataset, trains the MNISTNet model for a few epochs,
evaluates on the test set, and saves the trained weights to 'mnist_cnn.pth'.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MNISTNet


def train(model, device, train_loader, optimizer, loss_fn, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {running_loss / (batch_idx + 1):.4f} "
                f"Acc: {100. * correct / total:.2f}%"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    print(f"  Epoch {epoch} — Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model, device, test_loader, loss_fn):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.2f}%")
    return test_loss, accuracy


def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    MODEL_PATH = "mnist_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms: convert to tensor (scales to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load MNIST
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Initialize model, loss, optimizer
    model = MNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n--- Training ---")
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch)
        evaluate(model, device, test_loader, loss_fn)
        print()

    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
