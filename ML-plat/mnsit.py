import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load data
transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL image → tensor
    transforms.Normalize((0.1307,), (0.3081,))   # normalize with dataset mean/std
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

class CNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
      nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),)
      self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(3136, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))

   def forward(self,x):
      x = self.conv_layers(x)
      x = self.fc_layers(x)
      return x

# 2. Model, loss, optimizer
model     = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (output.argmax(1) == y_batch).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# 4. Evaluation loop
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            output  = model(X_batch)
            loss    = criterion(output, y_batch)
            total_loss += loss.item()
            correct    += (output.argmax(1) == y_batch).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# 5. Run training
for epoch in range(10):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch+1} | "
          f"Train Loss: {train_loss:.3f} Acc: {train_acc:.3f} | "
          f"Test Loss: {test_loss:.3f} Acc: {test_acc:.3f}")
