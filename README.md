Deep Neural Network Implementation
This Python script defines a class DeepNeuralNetwork that implements a deep neural network model. The network can be trained using gradient descent and supports various functionalities such as forward propagation, cost calculation, evaluation, training, saving/loading models, and one-hot encoding/decoding.

Installation
Ensure you have Python installed on your system. This implementation requires NumPy for numerical computations and Matplotlib for plotting during training.

pip install numpy matplotlib
Usage
To use this script, you need to create an instance of the DeepNeuralNetwork class, specifying the input dimension (nx) and the architecture of the network (list of layer sizes).

from deep_neural_network import DeepNeuralNetwork

# Initialize the network with 784 inputs and 2 hidden layers of sizes 128 and 64
nn = DeepNeuralNetwork(784, [128, 64])

# Train the network
nn.train(X_train, Y_train, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)

# Evaluate the network
accuracy, cost = nn.evaluate(X_test, Y_test)
print(f"Accuracy: {accuracy}, Final Cost: {cost}")
Replace X_train, Y_train, X_test, and Y_test with your actual data.

Methods
init(self, nx, lay): Initializes the neural network with the specified input dimension and architecture.
activation(x): Applies the sigmoid activation function to the input.
forward_prop(X): Performs forward propagation through the network.
cost(Y, A): Calculates the categorical cross-entropy cost between the true labels Y and the predicted activations A.
evaluate(X, Y): Evaluates the network on the provided data X and compares the predictions to the true labels Y. Returns accuracy and cost.
train(X, Y, iterations, alpha, verbose, graph, step): Trains the network for a specified number of iterations using gradient descent. Optionally plots the training cost over time.
save(filename): Saves the current state of the network to a pickle file.
load(filename): Loads a previously saved network state from a pickle file.
one_hot_encode(Y, classes): Encodes the target variable Y into one-hot vectors.
one_hot_decode(one_hot): Decodes one-hot encoded vectors back to original labels.
Notes
Ensure that the input data X is preprocessed appropriately (e.g., normalized).
The train method includes options for verbosity and plotting the training progress.
The evaluate method provides both the accuracy and the final cost of the model on the test set.
The save and load methods allow for persistence of trained models across sessions.
One-hot encoding and decoding are supported for handling multi-class classification problems.
This implementation is designed for educational purposes and may require adjustments for real-world applications, especially regarding data preprocessing and optimization parameters.
