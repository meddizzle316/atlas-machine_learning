### Deep Neural Network Implementation in Python
This repository contains a Python implementation of a deep neural network. The network is designed to handle classification problems and includes functionalities such as forward propagation, backward propagation, cost calculation, evaluation, training, saving/loading models, and one-hot encoding/decoding.

# Features
Deep Neural Network Class: Implements a deep neural network with customizable architecture.
Activation Functions: Uses sigmoid activation function for hidden layers and softmax for the output layer.
Forward Propagation: Computes the activations of all neurons in the network.
Backward Propagation: Updates the weights and biases based on the calculated gradients.
Cost Function: Utilizes categorical cross-entropy for multi-class classification problems.
Evaluation: Provides functionality to evaluate the model's performance on new data.
Training: Includes methods for training the model with gradient descent optimization.
Saving and Loading Models: Allows for saving and loading trained models using pickle.
One-Hot Encoding and Decoding: Supports one-hot encoding and decoding for input labels.
Installation
Ensure you have Python installed on your system. This project requires NumPy for numerical computations and Matplotlib for plotting (optional).

pip install numpy matplotlib
Usage
To use this deep neural network, create an instance of the DeepNeuralNetwork class, specifying the number of inputs (nx) and the structure of the network (lay). Then, train the network using the train method and evaluate its performance.

Example
from deep_neural_network import DeepNeuralNetwork

# Initialize the network
nn = DeepNeuralNetwork(nx=784, lay=[256, 128])

# Train the network
nn.train(X_train, Y_train, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)

# Evaluate the network
accuracy, cost = nn.evaluate(X_test, Y_test)
print(f"Accuracy: {accuracy}, Final Cost: {cost}")
Replace X_train, Y_train, X_test, and Y_test with your dataset.
