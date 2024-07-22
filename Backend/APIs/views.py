from rest_framework.decorators import api_view
from rest_framework.response import Response
from . import  Merged_predict_script,phcx_to_image,CNN_ONNX,ANN_ONNX
import numpy as np
import os
import json
from .tool import PulsarFeatureLab
import pandas as pd
import base64

@api_view(['POST'])
def cnn_predict(request):
    print(request.data)
    image_data = request.FILES['image'].read()
    model_path = os.path.join('models', 'cnn.onnx')  # Use the ONNX model
    output, probablity = CNN_ONNX.predict_image(CNN_ONNX.load_model(model_path), image_data)
    return Response({'prediction': int(output), 'probability': float(probablity)})

@api_view(['POST'])
def ann_predict(request):
    model_path = os.path.join('models', 'ann.onnx')  # ONNX model path
    # print("Request",request)
    print("Request Data",request.data)
    data = request.data['data']
    data = json.loads(data)  # Convert JSON data to a list of floats

    # Load the ONNX model
    ort_session = ANN_ONNX.load_model(model_path)

    # Perform prediction
    output, probability = ANN_ONNX.predict(ort_session, data)

    return Response({'prediction': int(output), 'probability': float(probability)})

@api_view(['POST'])
def merged_predict(request):
    print(request.data)
    try:
        ann_data = request.data['data']
        ann_data = json.loads(ann_data)
    except (KeyError, TypeError, ValueError):
        return Response({'error': 'Invalid ann_data'}, status=400)
    
    ann_data = np.array(ann_data).astype(np.float32)
    image_data = request.FILES['image'].read()

    ann_model_path = os.path.join('models', 'ann.onnx')
    cnn_model_path = os.path.join('models', 'cnn.onnx')

    ann_model = ANN_ONNX.load_model(ann_model_path)
    ann_output, ann_prob = ANN_ONNX.predict(ann_model, ann_data)

    cnn_model = CNN_ONNX.load_model(cnn_model_path)
    cnn_output, cnn_prob = CNN_ONNX.predict_image(cnn_model, image_data)

    merged_output, merged_prob = Merged_predict_script.get_merged_output(ann_prob, cnn_prob)

    return Response({
        'cnn_prediction': int(cnn_output),
        'cnn_probability': float(cnn_prob),
        'ann_prediction': int(ann_output),
        'ann_probability': float(ann_prob),
        'merged_prediction': int(merged_output),
        'merged_probability': float(merged_prob)
    })

@api_view(['POST'])
def phcx_predict(request):
    phcx_file = request.FILES['file']
    temporary_dir = os.path.join('temporary')
    temp_phcx_file_path = os.path.join(temporary_dir, 'temporary.phcx')
    with open(temp_phcx_file_path, 'wb') as temp_phcx_file:
        for chunk in phcx_file.chunks():
            temp_phcx_file.write(chunk)
    PulsarFeatureLab.run(temporary_dir)

    # csv_file_path = os.path.join(temporary_dir, 'output.csv')
    try:
        ann_data_df = pd.read_csv('temporary/output.csv',header=None)
        ann_data = ann_data_df.values.astype(np.float32)
        # print("ANN_DATA",ann_data)
    except Exception as e:
        return Response({'error': f'Error reading CSV file: {str(e)}'}, status=400)

    ann_model_path = os.path.join('models', 'ann.onnx')
    cnn_model_path = os.path.join('models', 'cnn.onnx')

    ann_model = ANN_ONNX.load_model(ann_model_path)
    ann_output, ann_prob = ANN_ONNX.predict(ann_model, ann_data)

    image_file_path = phcx_to_image.process_file(os.path.join(temporary_dir, 'temporary.phcx'))
    image_data = open(image_file_path, 'rb').read()

    cnn_output, cnn_prob = CNN_ONNX.predict_image(CNN_ONNX.load_model(cnn_model_path), image_data)

    merged_output, merged_prob = Merged_predict_script.get_merged_output(ann_prob, cnn_prob)

    image_base64 = base64.b64encode(image_data).decode('utf-8')
    print("Image Base64",image_base64)

    return Response({
        'cnn_prediction': int(cnn_output),
        'cnn_probability': float(cnn_prob),
        'ann_prediction': int(ann_output),
        'ann_probability': float(ann_prob),
        'merged_prediction': int(merged_output),
        'merged_probability': float(merged_prob),
        'generated_data': ann_data.tolist(),
        'image_base64': image_base64
    })
