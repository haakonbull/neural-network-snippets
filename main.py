# main.py
# This script trains a simple neural network from scratch on the MNIST dataset 
# and allows prediction on a given input image file.
# Comments are in English for clarity.

import numpy as np
from PIL import Image
import sys
import os
from tensorflow.keras.datasets import mnist



class NeuralNetwork:
    """Simple fully connected neural network with one hidden layer."""
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        """Initialize weights and biases."""
        # Use He initialization for ReLU (scaled normal distribution)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def train(self, X_train, y_train, epochs=5, batch_size=64, learning_rate=0.1):
        """Train the network using mini-batch gradient descent."""
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Shuffle the training data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                # Forward pass
                z1 = X_batch.dot(self.W1) + self.b1             # Hidden layer linear combination
                a1 = np.maximum(z1, 0)                         # ReLU activation
                z2 = a1.dot(self.W2) + self.b2                 # Output layer linear combination
                exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax
                # Compute cross-entropy loss for this batch
                correct_logprobs = -np.log(probs[range(len(X_batch)), y_batch])
                epoch_loss += np.sum(correct_logprobs)
                # Backpropagation
                dscores = probs
                dscores[range(len(X_batch)), y_batch] -= 1      # dL/dz2 for softmax
                dscores /= len(X_batch)
                # Gradient for W2 and b2
                dW2 = a1.T.dot(dscores)
                db2 = np.sum(dscores, axis=0)
                # Gradient for hidden layer
                dhidden = dscores.dot(self.W2.T)
                dhidden[z1 <= 0] = 0                            # Backprop through ReLU
                # Gradient for W1 and b1
                dW1 = X_batch.T.dot(dhidden)
                db1 = np.sum(dhidden, axis=0)
                # Update weights and biases
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
            # Print average loss for this epoch
            avg_loss = epoch_loss / n_samples
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X):
        """Predict class labels for input data."""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Forward pass (no training, just inference)
        z1 = X.dot(self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Return the predicted class indices
        return np.argmax(probs, axis=1)

def preprocess_image(image_path):
    """Load an image file and preprocess it into a 28x28 grayscale vector."""
    img = Image.open(image_path)
    img = img.convert('L')                          # convert to grayscale
    img = img.resize((28, 28))                      # resize to 28x28 pixels
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
    img_vector = img_array.flatten()                # flatten to 1D vector of length 784
    return img_vector

if __name__ == "__main__":
    # Load MNIST dataset (uses Keras API to download if not already available)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Preprocess dataset: flatten images and normalize pixel values
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    # Initialize and train the neural network
    net = NeuralNetwork(input_size=784, hidden_size=64, output_size=10)
    print("Training on MNIST dataset...")
    net.train(X_train, y_train, epochs=5, batch_size=64, learning_rate=0.1)
    # Evaluate the model on the test set
    test_preds = net.predict(X_test)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    # If an image file path is provided, preprocess and predict its digit
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        img_vector = preprocess_image(image_path)
        pred_label = net.predict(img_vector)[0]
        print(f"Predicted digit for image '{image_path}': {pred_label}")
