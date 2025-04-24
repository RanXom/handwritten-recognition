from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from PIL import Image
import io
from utils.preprocessing import preprocess_image
from model.mnist_model import make_prediction
import os

app = Flask(__name__, static_folder='../frontend/public')

# Load trained parameters
try:
    params = np.load('./model/trained_params.npz')
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    print("Model parameters loaded successfully!")
except Exception as e:
    print(f"Error loading model parameters: {e}")
    print("Please run train_model.py first to generate the model parameters")
    W1, b1, W2, b2 = None, None, None, None

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if W1 is None or b1 is None or W2 is None or b2 is None:
        return jsonify({'error': 'Model parameters not loaded. Run train_model.py first.'}), 500
    
    try:
        # Get image data from request
        image_data = request.json.get('image')
        
        # Decode base64 image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        processed_image = preprocess_image(img)
        
        # Make prediction
        prediction = int(make_prediction(processed_image, W1, b1, W2, b2)[0])
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
