import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def preprocess_image(image):
    """
    Preprocess the uploaded image to match MNIST format
    """
    # Save original image for debugging
    debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(debug_dir, 'original.png'), image)
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28 pixels
    image = cv2.resize(image, (28, 28))
    
    # Save resized image
    cv2.imwrite(os.path.join(debug_dir, 'resized.png'), image)
    
    # Invert image if needed (MNIST has white digits on black background)
    # Check if image needs inversion (is it black on white or white on black?)
    if np.mean(image) > 127:  # If average is bright, we need to invert
        image = 255 - image
    
    # Save inverted image
    cv2.imwrite(os.path.join(debug_dir, 'inverted.png'), image)
    
    # Normalize pixel values to [0, 1] as in the training
    image = image / 255.0
    
    # Save visualization of normalized image
    cv2.imwrite(os.path.join(debug_dir, 'normalized.png'), (image * 255).astype(np.uint8))
    
    # Reshape to match model input (784,1)
    img_vector = image.reshape(784, 1)
    
    return img_vector
