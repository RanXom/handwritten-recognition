import numpy as np

class MNISTModel:
    def __init__(self):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    def init_params(self):
        # Initialize parameters with proper scaling
        self.W1 = np.random.randn(10, 784) * 0.01
        self.b1 = np.zeros((10, 1))
        self.W2 = np.random.randn(10, 10) * 0.01
        self.b2 = np.zeros((10, 1))
        return self.W1, self.b1, self.W2, self.b2
    
    def ReLU(self, Z):
        return np.maximum(Z, 0)
    
    def softmax(self, Z):
        # Fix: return the softmax result
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtract max for numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2
    
    def one_hot_encode(self, Y):
        one_hot = np.zeros((Y.size, int(Y.max()) + 1))
        one_hot[np.arange(Y.size), Y.astype(int)] = 1
        one_hot = one_hot.T
        return one_hot
    
    def derivative_ReLU(self, Z):
        return Z > 0
    
    def back_propagation(self, Z1, A1, Z2, A2, X, Y):
        m = Y.size
        one_hot_Y = self.one_hot_encode(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)  # Fix: use axis=1 instead of 2
        dZ1 = self.W2.T.dot(dZ2) * self.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)  # Fix: use axis=1 instead of 2
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1  # Fix: update b1 instead of b2
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
    
    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)
    
    def get_accuracy(self, predictions, Y):
        return np.mean(predictions == Y)
    
    def gradient_descent(self, X, Y, iterations, alpha):
        self.init_params()
        
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(X)
            dW1, db1, dW2, db2 = self.back_propagation(Z1, A1, Z2, A2, X, Y)
            self.update_parameters(dW1, db1, dW2, db2, alpha)
            
            if i % 100 == 0:
                predictions = self.get_predictions(A2)
                accuracy = self.get_accuracy(predictions, Y)
                print(f"Iteration: {i}, Accuracy: {accuracy}")
        
        return self.W1, self.b1, self.W2, self.b2
    
    def save_model(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_model(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
    
    def predict(self, X):
        _, _, _, A2 = self.forward_propagation(X)
        predictions = self.get_predictions(A2)
        return predictions