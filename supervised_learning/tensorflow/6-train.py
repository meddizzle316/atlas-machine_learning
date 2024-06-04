#!/usr/bin/env python3
"""for stupid stupid tf1"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """builds trains and saves a neural network classifier"""

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    sess = tf.Session()
    y_pred = forward_prop(X_train, layer_sizes, activations)
    training_cost = calculate_loss(Y_train, y_pred)
    train_op = create_train_op(training_cost, alpha)

    init_vars = tf.global_variables_initializer()

    sess.run(init_vars)
        

    for iteration in range(iterations):
        # y_pred = forward_prop(X_train, layer_sizes, activations)
        # training_cost = calculate_loss(Y_train, y_pred)
        
        sess.run((train_op), feed_dict={x: X_train, y:Y_train})

        if iteration + 1 % 100 == 0:
            print(f"After {iteration} iterations:")
            # print(f"\tTraining Cost: {training_cost}")
            training_accuracy = calculate_accuracy(Y_train, y_pred)
            print(f"\tTraining Accuracy: {training_accuracy}")
            # print(f"\tValidation Cost: {validation_cost}")
            # print(f"\tValidation Accuracy: {validation_accuracy}")
