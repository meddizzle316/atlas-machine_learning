# Deep Neural Network Implementation
This Python script defines a class DeepNeuralNetwork that implements a deep neural network model. The network can be trained using gradient descent and supports various functionalities such as forward propagation, cost calculation, evaluation, training, saving/loading models, and one-hot encoding/decoding.

#### Installation
Ensure you have Python installed on your system. This implementation requires NumPy for numerical computations and Matplotlib for plotting during training.

``` pip install numpy matplotlib ```

### Usage
To use this script, you will need to create an instance of the DeepNeuralNetwork class, configure its architecture, and then train it with your data.

#### Initialization
```from deep_neural_network import DeepNeuralNetwork```

Initialize the network with input dimension and layer sizes
```nn = DeepNeuralNetwork(input_dim=784, layer_sizes=[128, 64, 10])```


# Training
Before training, ensure your input data X and target labels Y are preprocessed appropriately. For example, normalize X and encode Y using one-hot encoding.

# Example training data
```X_train = ... # Your input data here```

```Y_train = ... # Your target labels here```

# Train the network
```nn.train(X_train, Y_train, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100)```

# Evaluation
After training, you can evaluate the network's performance on new data.

#### Example evaluation data
```X_test = ... # Your test input data here```

```Y_test = ... # Your test target labels here```

### Evaluate the network
```accuracy, cost = nn.evaluate(X_test, Y_test)```

```print(f"Accuracy: {accuracy}, Final Cost: {cost}")```

### Saving and Loading Models
You can save the trained model to a file and load it later without retraining.

#### Save the model
```nn.save('model.pkl')```

#### Load the model
```loaded_model = DeepNeuralNetwork.load('model.pkl')```

### Note
This implementation assumes that the input data X is normalized to have zero mean and unit variance, and the target labels Y are one-hot encoded. Adjustments may be necessary depending on your specific dataset and problem requirements.
