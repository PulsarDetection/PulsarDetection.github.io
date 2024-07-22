# ANN_predict_script.py
import numpy as np
import onnxruntime as ort

def load_model(model_path):
    # Load the ONNX model
    ort_session = ort.InferenceSession(model_path)
    return ort_session

def predict(ort_session, X):
    # Convert the input to the expected format
    X = np.array(X, dtype=np.float32).reshape(1, -1)  # Reshape to match the input shape
    inputs = {ort_session.get_inputs()[0].name: X}
    outputs = ort_session.run(None, inputs)
    probability = outputs[0][0][0]  # Assuming the output is of shape (1, 1)

    if probability > 0.5:
        output = 1
        if probability > 1:
            probability = 1
    else:
        output = 0
        if probability < 0:
            probability = 0

    return output, probability
