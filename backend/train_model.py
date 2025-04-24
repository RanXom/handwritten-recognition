
import numpy as np
import os
from model.mnist_model import MNISTModel

# Load MNIST data (assuming you have the data files)
# In a real deployment, we'd save these steps

def load_data():
    try:
        # Try to load data from local directory
        data = np.load(os.path.join(os.path.dirname(__file__), 'data', 'mnist_sample.npz'))
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
    except:
        # If not available, use a dummy small dataset for testing
        print("Using dummy data for testing")
        X_train = np.random.rand(784, 1000)
        Y_train = np.random.randint(0, 10, 1000)
        X_test = np.random.rand(784, 100)
        Y_test = np.random.randint(0, 10, 100)
    
    return X_train, Y_train, X_test, Y_test

def train():
    # Load data
    X_train, Y_train, X_test, Y_test = load_data()
    
    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Initialize and train model
    model = MNISTModel()
    W1, b1, W2, b2 = model.gradient_descent(X_train, Y_train, iterations=1000, alpha=0.1)
    
    # Test accuracy
    _, _, _, A2 = model.forward_propagation(X_test)
    predictions = model.get_predictions(A2)
    accuracy = model.get_accuracy(predictions, Y_test)
    print(f"Test accuracy: {accuracy}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'trained_params.npz')
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()