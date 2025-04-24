from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from PIL import Image
import io
from utils.preprocessing import preprocess_image
from model.mnist_model import make_prediction, forward_propogation, get_predictions
import os

app = Flask(__name__, static_folder='../frontend/public', static_url_path='')

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

# Debugging code - Test with MNIST Test dataset
def test_with_mnist():
    import pandas as pd
    try:
        test_data = pd.read_csv('./datasets/mnist_test.csv')
        # Take the first 10 test samples
        test_samples = test_data.iloc[:10].values
        
        for i in range(10):
            # Get label and image
            label = test_samples[i, 0]
            image = test_samples[i, 1:].reshape(784, 1) / 255.0
            
            # Make prediction
            _, _, _, A2 = forward_propogation(W1, b1, W2, b2, image)
            prediction = get_predictions(A2)[0]
            
            print(f"MNIST Test {i}: True label={label}, Predicted={prediction}")
    except Exception as e:
        print(f"Error testing MNIST samples: {e}")

if W1 is not None:
    test_with_mnist()

if W1 is not None:
    print(f"W1 shape: {W1.shape}")
    print(f"b1 shape: {b1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"b2 shape: {b2.shape}")
    
    # Check if model is extremely biased
    test_input = np.zeros((784, 1))
    preds = []
    for i in range(10):
        test_input[i*78] = 1.0  # Just add a single pixel
        _, _, _, A2 = forward_propogation(W1, b1, W2, b2, test_input)
        pred = get_predictions(A2)[0]
        preds.append(int(pred))
        test_input[i*78] = 0.0  # Reset
    
    print(f"Test predictions for single-pixel inputs: {preds}")

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
        
        # Make prediction with detailed output
        Z1, A1, Z2, A2 = forward_propogation(W1, b1, W2, b2, processed_image)
        prediction = int(get_predictions(A2)[0])
        
        # Log the softmax output probabilities
        probabilities = A2.reshape(-1).tolist()
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
