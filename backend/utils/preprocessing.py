import numpy as np
import cv2

def preprocess_image(image):
    """
    Preprocess uploaded image to match MNIST format
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    
    # Invert colors (MNIST has white digits on black background)
    inverted = cv2.bitwise_not(resized)
    
    # Normalize pixel values to [0, 1]
    normalized = inverted / 255.0
    
    # Reshape to match model input (784,)
    flattened = normalized.reshape(784, 1)
    
    return flattened