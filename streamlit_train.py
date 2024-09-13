# Complete logistic_regression_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.optim as optim
import os


# Define the Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  # 784 input features and 10 output classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        return self.linear(x)  # Return raw logits


# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into training and validation sets
train_data, val_data = random_split(train_dataset, [48000, 12000])

# DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


def train_model(device, num_epochs=12, learning_rate=0.01, weight_decay=1e-5,
                save_path='best_logistic_regression_model.pth'):
    # Initialize the model, loss function, and optimizer
    model = LogisticRegressionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        train_acc = correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()

        val_acc = correct_val / total_val
        avg_val_loss = running_val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f"Saved Best Model with Val Acc: {best_val_acc:.4f}\n")

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


if __name__ == "__main__":
    # Determine the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the model
    trained_model, best_val_acc = train_model(device=device, num_epochs=12, learning_rate=0.01, weight_decay=1e-5,
                                              save_path='best.pth')
