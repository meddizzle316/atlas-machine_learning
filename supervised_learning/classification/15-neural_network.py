#!/usr/bin/env python3
"""defines a Neural Network class"""
import numpy as np


class NeuralNetwork:
    """class for Neural Network"""

    def __init__(self, nx, nodes):
        """init"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # weights vector - hidden layer. Random initialization
        self.__W1 = np.random.normal(np.zeros((nodes, nx)))

        # bias -- hidden layer. initial value = 0
        self.__b1 = np.zeros((nodes, 1))

        # activated output - hidden layer. initial value = 0
        self.__A1 = 0

        # weights vector - output neuron. Random initialization
        self.__W2 = np.random.normal(np.zeros((1, nodes)))

        # bais -- output neuron. initial value = 0
        self.__b2 = 0

        # Activated output -- output neuron (prediction). initial = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2

    def forward_prop(self, X):
        """forward propogation for neural network"""
        self.__A1 = self.activation(np.dot(self.W1, X) + self.b1)
        self.__A2 = self.activation(np.dot(self.W2, self.A1) + self.b2)

        return (self.__A1, self.__A2)

    def activation(self, x):
        """sigmoid activation function"""
        return 1 / (1 + (np.exp(-x)))

    def cost(self, Y, A):
        """calculates cost of model using logistic regression"""
        """Y: target or correct values"""
        """A: real values or activated output of the neuron"""
        firstPart = Y * np.log(A)
        secondPart = ((1 - Y) * np.log(1.0000001 - A))
        return np.mean(-(np.mean(firstPart) + (secondPart)))

    def evaluate(self, X, Y):
        """evaluates the neuron's predictions"""
        """Y: target labels"""
        """X: input data"""
        pred1, pred2 = self.forward_prop(X)
        # I guess prediction just means the dotprod plust bias?
        new_out = np.where(pred2 >= 0.5, 1, 0)
        return (new_out), self.cost(Y, pred2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient descent with 1 hidden layers"""
        # getting number of training examples (m)
        m = X.shape[1]

        # hidden layer
        dz2 = A2 - Y

        dw2 = np.matmul(dz2, A1.T) / m

        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        z1 = np.matmul(self.__W1, X) + self.b1

        # first layer
        dz1 = ((self.__W2.T * dz2)) * (A1 * (1 - A1))
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        dw1 = np.matmul(dz1, X.T) / m

        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """trains the neural network"""
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
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            count += 1
            if verbose == True and count >= step:
                count = 0
                A, cost = self.evaluate(X, Y)
                costList.append(cost)
                iterationsList.append((iteration + 1))
                print(f"Cost after {iteration + 1} iterations: {cost}")
        if graph == True:
            plt.plot(iterationsList, costList)
            # plt.xticks(np.linspace(0, iterations, int(iterations / 500) + 1))
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("Training Cost")
        return self.evaluate(X, Y)
