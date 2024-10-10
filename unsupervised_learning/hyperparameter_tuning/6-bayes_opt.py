#!/usr/bin/env python3
"""
optimizes machine learning model of my choice
using GPyOpt

Here are the requirements
Your script should optimize at least 5 different hyperparameters.
    E.g. learning rate, number of units in a layer, dropout rate,
    L2 regularization weight, batch size
Your model should be optimized on a single satisfying metric
Your model should save a checkpoint of its best iteration during
each training session
The filename of the checkpoint should specify the values of
the hyperparameters being tuned
Your model should perform early stopping
Bayesian optimization should run for a maximum of 30 iterations
Once optimization has been performed, your script should
plot the convergence
Your script should save a report of the optimization to
the file 'bayes_opt.txt'
There are no restrictions on imports
"""
# %%
import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt


x, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.fit_transform(x_val)



def get_optimizer(case, learning_rate):
    """because GPyOpt doesn't take strings
    we have to jankily convert the integers
    to the proper tensorflow optimizer"""
    if case == 0:
        return Adam(learning_rate=learning_rate)
    if case == 1:
        return Adagrad(learning_rate=learning_rate)
    if case == 2:
        return SGD(learning_rate=learning_rate)



def create_model(hyperparameters):
    num_neurons = int(hyperparameters[0][0])
    learning_rate = hyperparameters[0][1]
    num_layers = int(hyperparameters[0][2])
    optimizer_case = int(hyperparameters[0][3])

    model = Sequential()
    for i in range(num_layers):
        model.add(Dense(num_neurons, input_dim=20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = get_optimizer(optimizer_case, learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model



def keras_model_validation_loss(hyperparameters):
    model = create_model(hyperparameters)
    batch_size = int(hyperparameters[0][4])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')

    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val),
                        callbacks=[early_stopping])

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=1)

    return val_loss



bounds = [
    {'name': 'num_neurons', 'type': 'discrete', 'domain': np.arange(10, 101, 10)},
    {'name': 'learning_rate', 'type': "continuous", 'domain': (1e-4, 1e-1)},
    {'name': 'num_layers', 'type': "discrete", 'domain': (1, 10)},
    {'name': 'optimizer', 'type': "discrete", 'domain': (0, 1, 2)},
    {'name': 'batch_size', 'type': "discrete", 'domain': (8, 16, 32, 64, 128, 512)},
]

optimizer = GPyOpt.methods.BayesianOptimization(
    f=keras_model_validation_loss,
    domain=bounds,
    acquisition_type='EI',
    maximize=False
)

optimizer.run_optimization(max_iter=2)

print("Optimal number of neurons", optimizer.x_opt[0])
print("optimal learning rate", optimizer.x_opt[1])
print("optimal number of layers", optimizer.x_opt[2])

if optimizer.x_opt[3] == 0:
    opt_alg = 'Adam'
if optimizer.x_opt[3] == 1:
    opt_alg = 'Adagrad'
if optimizer.x_opt[3] == 2:
    opt_alg = 'SGD'

print("optimal optimizer", opt_alg)
print("optimal batch_size", optimizer.x_opt[4])
print("best validation loss", optimizer.fx_opt)

with open('bayes_opt.txt', "w") as file:
    index = 0
    for opt, bound in zip(optimizer.x_opt, bounds):
        if index != 3:
            file.write(f"Optimal {bound['name']}: {np.array2string(opt)}\n")
        else:
            file.write(f"Optimal {bound['name']}: {opt_alg}\n")
        index += 1
    file.write(f"best validation loss: {np.array2string(optimizer.fx_opt)}")

iterations = np.arange(len(optimizer.Y))
values = optimizer.Y.flatten()
plt.plot(iterations, values, label='Optimal')
plt.show()



