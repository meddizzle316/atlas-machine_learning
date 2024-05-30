#!/usr/bin/env python3
"""defines a deep neural network"""
import numpy as np
import pickle

class DeepNeuralNetwork():
    """class for deepneuralnetwork"""
    def __init__(self, nx, lay):
        """init func"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(lay, list):
            raise TypeError("layers must be a list of positive integers")

        test_array = np.array([lay])
        if not np.all(test_array > 0) or len(lay) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(lay)
        self.__cache = {}
        self.__weights = {}

        # setting dimensions and initialization of first layer
        self.__weights[f"W{1}"] = np.random.randn(lay[0], nx) * np.sqrt(2 / nx)
        self.__weights[f"b{1}"] = np.zeros((lay[0], 1))

        # setting dimensions and initialization of 2nd to the last lay
        for i in range(1, self.L):
            # initialize using He et al method (mostly the sqrt at the end?)
            # X and rand: dumb pycode format fix line 35
            X = np.sqrt(2. / lay[i - 1])
            rand = np.random.randn(lay[i], lay[i - 1])
            self.__weights[f"W{i + 1}"] = rand * X
            self.__weights[f"b{i + 1}"] = np.zeros((lay[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
    
    def activation(self, x):
        """sigmoid activation function"""
        return 1 / (1 + (np.exp(-x)))

    def forward_prop(self, X):
        """forward propr"""
        # needs to be modified for one_hot
        self.__cache[f"A{0}"] = X
        self.__cache[f"A{1}"] = self.activation(np.dot(self.weights[f'W{1}'], X) + self.weights[f'b{1}'])
        for i in range(2, self.L + 1):
            self.__cache[f"A{i}"] = self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}'])
        return (self.__cache[f"A{self.L}"], self.__cache)

    def cost(self, Y, A):
        """cost function"""
        # needs to be modified with one_hot
        firstPart = Y * np.log(A)
        secondPart = ((1 - Y) * np.log(1.0000001 - A))
        return np.mean(-(np.mean(firstPart) + (secondPart)))

    def evaluate(self, X, Y):
        """evaluate function"""
        output, cache = self.forward_prop(X)
        mod_output = np.where(output >= 0.5, 1, 0)
        # mod_output should be A_one_hot encoded
        return (mod_output, self.cost(Y, output))
    
    def activationDerivative(self, x):
        """sigmoid activation derivative"""
        return x * (1 - x)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """performs gradient descent"""

        # getting total number of 'examples'
        m = Y.shape[1]
        
        for layer in range(self.L, 0, -1):
            # going backwards through layers
            if layer == self.L:
                # dz is different in first layer going backwards
                a = self.__cache[f"A{layer}"]
                dz = a - Y
            else:
                # dz is this every iteration besides first
                a = self.__cache[f"A{layer}"]

                # da is from last iteration
                dz = da * self.activationDerivative(a)

            # dynamically calculates derivatives in each layer
            dw = np.matmul(dz, self.__cache[f"A{layer - 1}"].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da = np.matmul(self.__weights[f"W{layer}"].T, dz)
        
            # update weights dictionary (weights and bias)
            W = self.__weights[f'W{layer}']
            b = self.__weights[f'b{layer}']
            self.__weights[f'W{layer}'] = W - (alpha * dw)
            self.__weights[f'b{layer}'] = b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the deep neural network"""
        import matplotlib.pyplot as plt

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        count:int = 0
        costList = []
        A, cost = self.evaluate(X, Y)
        costList.append(cost)
        iterationsList = []
        iterationsList.append(0)
        
        for iteration in range(iterations):
            output, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            count += 1
            if verbose == True and count >= step:
                count = 0
                A, cost = self.evaluate(X, Y)
                costList.append(cost)
                iterationsList.append(iteration)
                print(f"Cost after {iteration + 1} iterations: {cost}")

        if graph == True:
            plt.plot(iterationsList, costList)
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("Training Cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """saves instance to pickle format file"""
        try:
            filename.split('.')[1]
            with open(f"{filename}", 'wb') as file:
                pickle.dump(self, file)
        except IndexError:
            with open(f"{filename}.pkl", 'wb') as file:
                pickle.dump(self, file)

            
    def load(filename):
        """loads instance from pickle format file"""
        try:
            with open(filename, 'rb') as file:
                loaded_data = pickle.load(file)
        except FileNotFoundError:
            return None
        return loaded_data
