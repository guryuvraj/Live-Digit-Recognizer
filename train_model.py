# train_model.py

import torch
import torchvision
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
n_epochs = 10
batch_size_train = 200
batch_size_validation = 5000
batch_size_test = 1000
learning_rate = 0.01  # Adjusted for SGD
momentum = 0.9        # Added momentum for SGD
log_interval = 100

# Seed and Device Configuration
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Model
class MultipleLinearRegression(nn.Module):
    def __init__(self):
        super(MultipleLinearRegression, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)             # Raw logits
        return x

# Initialize Dataset and DataLoaders
MNIST_training = torchvision.datasets.MNIST('./MNIST_dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ]))

MNIST_test_set = torchvision.datasets.MNIST('./MNIST_dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ]))

# Split into training and validation sets
MNIST_training_set, MNIST_validation_set = random_split(MNIST_training, [55000, 5000])

train_loader = DataLoader(MNIST_training_set, batch_size=batch_size_train, shuffle=True)
validation_loader = DataLoader(MNIST_validation_set, batch_size=batch_size_validation, shuffle=False)
test_loader = DataLoader(MNIST_test_set, batch_size=batch_size_test, shuffle=False)

# Initialize the Model, Optimizer
multi_linear_model = MultipleLinearRegression().to(device)
optimizer = optim.SGD(multi_linear_model.parameters(), lr=learning_rate, momentum=momentum)

# Training Function
def train(epoch, data_loader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)               # Raw logits
        loss = F.cross_entropy(output, target)  # Cross-Entropy applies Softmax internally
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} '
                  f'({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluation Function
def evaluate(data_loader, model, dataset_name):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)                       # Raw logits
            loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum loss over batch
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logit
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print(f'{dataset_name} set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return accuracy

# Training Loop
best_accuracy = 0.0
best_epoch = 0

for epoch in range(1, n_epochs + 1):
    train(epoch, train_loader, multi_linear_model, optimizer)
    validation_accuracy = evaluate(validation_loader, multi_linear_model, "Validation")
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_epoch = epoch
        torch.save(multi_linear_model.state_dict(), "best_model.pth")
        print(f"New best model saved at epoch {epoch} with validation accuracy {validation_accuracy:.2f}%\n")

# Load Best Model for Final Testing
best_model = MultipleLinearRegression().to(device)
best_model.load_state_dict(torch.load("best_model.pth", map_location=device))
test_accuracy = evaluate(test_loader, best_model, "Test")
print(f"Best Model found at epoch {best_epoch} with Validation Accuracy: {best_accuracy:.2f}%")
print(f"Test set accuracy of the best model: {test_accuracy:.2f}%")
