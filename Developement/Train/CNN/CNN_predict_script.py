import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from torch import nn

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure image is resized to 32x32
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with your model's stats
])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def load_model(model_path):
    model = SimpleCNN()  # Define the model architecture
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Load the model's state_dict
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).numpy()
        prediction = (probabilities > 0.5).astype(int)  # Binary classification

    return prediction[0][0], probabilities[0][0]

def predict_images_from_dir(model_path, image_dir):
    model = load_model(model_path)
    predictions = {}
    
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            prediction, probability = predict_image(model, image_path)
            predictions[image_file] = {'prediction': int(prediction), 'probability': float(probability)}

    return predictions

# Hardcoded paths
model_path = 'CNN_model.pth' # give model path
image_path = 'saved_images/sample.jpeg' # give image path

# model loading
model = load_model(model_path)

# Perform predictions
output, probablity = predict_image(model, image_path)

# Print result
print(f"Prediction: {output}, Probability: {probablity}")
