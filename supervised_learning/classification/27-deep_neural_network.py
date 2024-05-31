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
            
            if i == self.L:
                z = np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}']
                t = np.exp(z)
                print("The result of the forward propagation with sigmoid")
                print(self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}']))
                print("this is the shape of the forward prop sigmoid result")
                # ask Isaac: Why couldn't we just run argmax axis=0 on the sigmoid? Shouldn't it 
                print(print(self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}']).shape))
                # applying softmax activation function to convert raw output 
                print("this is me seeing what would happen if we added the argmax function to the sigmoid output")
                print(np.argmax(self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}']), axis=0))
                # output [3 0 4 ... 8 4 8] which appears to be the same if you do that for the softmax function
                # which means there was a different reason for doing it
                # I guess it was just because we had to do operations with A3 and Y in the cost function? And since Y was 
                # now a one_hot encoded vector we needed to have an output that would work with it?
                self.__cache[f"A{i}"] = t/np.sum(t, axis=0)
                print("The result of the forward propagation with softmax")
                print(self.__cache[f"A{i}"])
                print("this is the shape of the forward prop result")
                print(self.__cache[f"A{i}"].shape)
            else:
                self.__cache[f"A{i}"] = self.activation(np.dot(self.weights[f'W{i}'], self.__cache[f"A{i - 1}"]) + self.weights[f'b{i}'])

        return (self.__cache[f"A{self.L}"], self.__cache)

    def cost(self, Y, A):
        """cost function"""
        # version of the cross entropy function 
        # that's called "categorical cross entropy"
        # and is used when the number of "classes"
        # or possible outputs is more than 2 (2 outputs is binary)
        m = Y.shape[1]
        print(f"this is the y one_hot {Y}")
        # shape of Y is (10, 50000)
        # Y looks like [[0. 1. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 0. 0. 0.]
        #  ...
        #  [0. 0. 0. ... 0. 0. 0.]
        #  [0. 0. 0. ... 1. 0. 1.]
        #  [0. 0. 0. ... 0. 0. 0.]]
        # shape of A is (10, 50000)
        # A looks like 
        #         [[7.82847419e-02 9.79862566e-01 1.82702610e-03 ... 7.32722429e-04
        #   3.35922141e-04 1.69250618e-04]
        #  [1.02834351e-03 8.49814019e-07 8.47756242e-04 ... 5.91432716e-02
        #   9.23885795e-04 6.05646821e-02]
        #  [9.25449456e-03 1.91277345e-04 1.73405778e-02 ... 3.88905417e-02
        #   3.70429825e-02 6.35733856e-03]
        #  ...
        #  [8.67839998e-03 1.10132402e-04 2.70420187e-02 ... 5.32493355e-04
        #   4.10496745e-03 7.44992083e-03]
        #  [1.78584601e-02 1.50456412e-04 1.02080091e-02 ... 5.55986477e-01
        #   1.32962306e-01 6.29559485e-01]
        #  [2.87492574e-03 2.22672137e-05 1.68391214e-01 ... 2.64717613e-03
        #   2.52859142e-01 5.27633619e-02]]
        print(f"this is the result of the cost function {np.sum(-Y * np.log(A)) / m}")
        return np.sum(-Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """evaluate function"""
        # get A3 or output layer, one that uses Softmax
        a = self.forward_prop(X)[0]

        # converts outputlayer from softmax into to a form where each neuron's activation 
        # corresponds to the probability of the class it represents
        a_ohe = np.argmax(self.__cache[f"A{self.L}"], axis=0)
        # what does this look like?
       
        # does that mean a_ohe is being prepared to do one_hot encoding
        print(f"this is a3 output with softmax normalization removed {a_ohe}")
        # a_ohe looks liks [3 0 4 ... 8 4 8]
        # so in this case, because we did argmax, axis=0, we basically went through
        # each column in the a3 output and returned the index with the highest value
        # which in our case, I think also corresponds to the number the nn
        # is currently predicting 
        print(f"this is the a_ohe shape before resizing{a_ohe.shape}")
        a_ohe.reshape(a_ohe.size, 1)
        print(f"this is the a_ohe shape after resizing{a_ohe.shape}")
        count = np.arange(a.shape[1])
        # like the one_hot encoder, this just returns an array of ascending values 
        # like [0, 1, 2, 3] for the size of a.shape[1] to give us one giant vector of (I think)
        # 50000 length (if a is (10, 50000))
        # what is the point of this, I'm getting 50000, for before and after
        count.reshape(count.size, 1)
        # I commented the above line ^ out and it didn't change anything?? I'll keep
        # it for reference but it seems like it was just a precaution
        hard_max = np.zeros_like(a)
        print(f"this is the hard_max before one_hot encoding{hard_max}")
        
        # is this basically one hot encoding the a softmax output?
        hard_max[a_ohe, count] = 1
        print(f"this is the hard_max after one_hot encoding{hard_max}")
        
        # mod_output = np.where(output >= 0.5, 1, 0)
        # mod_output should be A_one_hot encoded
        return (hard_max.astype(int), self.cost(Y, self.__cache[f"A{self.L}"]))
    
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
                # this operation works with both activation functions and 
                # even if Y is not one hot encoded (though to be fair, it was always
                # binary, just with 2 classes possible )
            else:
                # dz is this every iteration besides first
                a = self.__cache[f"A{layer}"]

                # da is from last iteration
                dz = da * self.activationDerivative(a)
                # if we use the sigmoid function until the very last one, why do we not need a softmax equivalent
                # I guess we would, except for this particular cross entropy error function (because gradient is derror/dweights)
                # we don't need the activation derivative 
                # the activation derivative would probably have to change if we used it for any other layer 
                # besides the last but since we don't, we don't have to mess with it
                # maybe ask Isaac? (or trying the last task and seeing what parts I have to change)

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

    def one_hot_encode(self, Y, classes):
        """encoding function"""
        if type(Y) is not np.ndarray or not isinstance(classes, int):
            return None
        if classes < 2 or classes < np.amax(Y):
            return None
        encoded_data = np.zeros((classes, len(Y)))

        encoded_data[Y, np.arange(Y.size)] = 1

        return encoded_data
    
    def one_hot_decode(self, one_hot):
        """decodes one hot"""
        import numpy as np
        
        if type(one_hot) is not np.ndarray:
            return None
        if len(one_hot) < 2:
            return None
        if one_hot.ndim != 2:
            return None
        decoded_data = []
        decoded_data = np.argmax(one_hot, axis=0)
        # decoded_data = np.append(decoded_data, decoded_data[0])
        # decoded_data = np.delete(decoded_data, 0)
        return decoded_data

