import torch
import torch.onnx
from torch import nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        layers = []
        layers.append(nn.Linear(8, 16))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 16))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(16, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

model = ANN()  # Define the model architecture
model.load_state_dict(torch.load('ANN_model.pth'))  # Load the model's state_dict
model.eval()
# Assuming you have your trained PyTorch model
x = torch.randn(1, 8)

torch.onnx.export(model, x, "model.onnx")