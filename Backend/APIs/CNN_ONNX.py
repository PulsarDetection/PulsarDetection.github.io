# CNN_predict_script.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure image is resized to 32x32
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with your model's stats
])

def load_model(model_path):
    ort_session = ort.InferenceSession(model_path)
    return ort_session

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_image(ort_session, image_data):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_data)).convert('RGB')  # Ensure image is RGB
    image = transform(image).unsqueeze(0).numpy()  # Add batch dimension

    # Run inference
    outputs = ort_session.run(None, {"input.1": image})
    probabilities = sigmoid(outputs[0])
    prediction = (probabilities > 0.5).astype(int)  # Binary classification
    
    return prediction[0][0], probabilities[0][0]
