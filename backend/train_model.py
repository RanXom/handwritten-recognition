import numpy as np
import pandas as pd
from model.mnist_model import gradient_descent

def train_and_save_model():
    # Load data
    print("Loading training data...")
    train_data = pd.read_csv('./datasets/mnist_train.csv')
    
    # Prepare data
    print("Preparing data...")
    data = np.array(train_data)
    m, n = data.shape
    np.random.shuffle(data)
    
    # Using all data for training (no validation split for final model)
    data_train = data.T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255  # Normalizing pixel values
    _, m_train = X_train.shape
    
    print(f"Training with {m_train} samples...")
    
    # Train the model
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)
    
    # Save the trained parameters
    print("Saving model parameters...")
    np.savez('./model/trained_params.npz', W1=W1, b1=b1, W2=W2, b2=b2)
    
    print("Model training complete and parameters saved!")

if __name__ == "__main__":
    train_and_save_model()
