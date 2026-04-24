import torch 
import torch.nn as nn


class CNN(nn.Module):
   def __init__(self):
      super().__init__()
      
      self.conv_layers = nn.Sequential(nn.Conv2d(1, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
      nn.Conv2d(32, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2))

      self.fc_layers = nn.Sequential(nn.Flatten(), nn.Linear(3136, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))

   def forward(self,x): 
      x = self.conv_layers(x)
      x = self.fc_layers(x)
      return x

model = CNN()
print(model)
