# Handwriting Recognition System

This is a handwriting recognition system that uses a neural network trained on the MNIST dataset to recognize handwritten digits from 0-9.

## Project Structure

```
handwriting-recognition/
├── frontend/              # Frontend code
│   ├── public/            # Static files
│   │   ├── index.html     # Main HTML page
│   │   ├── style.css      # CSS styles
│   │   └── assets/        # Images and other assets
│   └── src/
│       └── app.js         # Frontend JavaScript
├── backend/               # Backend code
│   ├── app.py             # Main Flask application
│   ├── model/             # Neural network model
│   │   ├── __init__.py    # Init File
│   │   ├── mnist_model.py # Neural network implementation
│   │   └── trained_params.npz # Pre-trained model parameters (generated)
│   ├── datasets/          # Training and test datasets
│   │   ├── mnist_train.csv
│   │   └── mnist_test.csv
│   ├── notebook/
│   │   └── Handwriting_Recognition.ipynb # Original notebook
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   └── preprocessing.py # Image preprocessing
│   ├── requirements.txt   # Python dependencies
│   ├── train_model.py     # Script to train and save model
│   └── Dockerfile         # For containerization
└── README.md              # This file
```

## Setup and Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Backend Setup
1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the model (this needs to be done only once):
   ```
   python train_model.py
   ```

4. Start the Flask server:
   ```
   python app.py
   ```

### Frontend
The frontend is served by the Flask application. Once the backend server is running, 
you can access the application by navigating to:
```
http://localhost:5000
```

## Using the Application
1. Draw a digit (0-9) on the canvas
2. Click the "Predict" button to see the recognition result
3. Alternatively, upload an image of a handwritten digit

## Docker Setup (Optional)
If you prefer to use Docker:

1. Build the Docker image:
   ```
   docker build -t handwriting-recognition -f backend/Dockerfile .
   ```

2. Run the Docker container:
   ```
   docker run -p 5000:5000 handwriting-recognition
   ```

3. Access the application at:
   ```
   http://localhost:5000
   ```

## Model Details
- The neural network is implemented from scratch using NumPy
- The model consists of a 2-layer neural network with ReLU and Softmax activations
- The model achieves approximately 85.4% accuracy on the validation set
