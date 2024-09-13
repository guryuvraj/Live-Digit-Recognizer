import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.optim as optim  # Import the optim module
import itertools

# from train_model import random_seed
#

def logistic_regression(device):
    """
    Implements logistic regression on the MNIST dataset.
    """

    # Logistic Regression Model
    class LogisticRegressionModel(nn.Module):
        def __init__(self):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(28 * 28, 10)  # 784 input features (28x28 image) and 10 output classes

        def forward(self, x):
            x = x.view(-1, 28 * 28)  # Flatten the image to a vector of 784 elements
            return self.linear(x)  # Linear layer for logistic regression

    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)
    # Data pipeline for training and validation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    # Load the dataset and split into training and validation sets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_data, val_data = torch.utils.data.random_split(train_dataset, [48000, 12000])

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = LogisticRegressionModel().to(device)
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)  # SGD optimizer with L2 regularization (weight_decay)

    num_epochs = 12
    best_val_acc = 0  # To keep track of the best validation accuracy

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Train the model
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero out the gradients
            outputs = model(data)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate the loss
            _, predicted = outputs.max(1)  # Get the index of the max log-probability (predicted class)
            total += targets.size(0)  # Update total number of examples
            correct += predicted.eq(targets).sum().item()  # Update number of correct predictions

        train_acc = correct / total  # Compute training accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_acc:.4f}')

        # Validation step
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation during validation
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)  # Get the index of the max log-probability (predicted class)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()  # Count correct predictions

        val_acc = correct / total  # Compute validation accuracy
        print(f'Validation Accuracy: {val_acc:.4f}')

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    # Return the best model after training
    results = dict(
        model=best_model
    )

    return results