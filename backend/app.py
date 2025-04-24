from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import image
import os

from model.mnist_model import MNISTModel
from utils.preprocessing import preprocess_image

app = Flask(__name__)
CORS(app)   # Enable CORS for all routes

# Initialize and load model
model = MNISTModel()
model_path = os.path.join(os.path.dirname(__file__), 'model', 'trained_params.npz')

# Check if trained model exists otherwise train a new one
if os.path.exists(model_path):
    model.load_model(model_path)
    print("Loaded pre-trained model")
else:
    print("No pre-trained model found. Please run training script first.")

@app.route('/predict', method=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image data from request
        image_data = request.json['image']

        # Remove data URL prefix if present
        if 'data:image' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Preprocess image
        processed_image = preprocess_image(image_np)

        # Make prediction
        prediction = int(model.predict(processed_image)[0])

        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ = '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
