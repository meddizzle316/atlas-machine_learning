#!/usr/bin/env python3
"""creates an autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder model
    returns
    encoder
    decoder
    auto -- the full model"""

    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = keras.layers.Dense(hidden_layers[0], activation='relu')(encoder_input)

    for num_nodes in range(1, len(hidden_layers)):
        x = keras.layers.Dense(hidden_layers[num_nodes], activation='relu')(x)
    x = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.models.Model(inputs=encoder_input, outputs=x)

    # decoder_input = keras.layers.Input(shape=(latent_dims,))
    encoder_output = encoder.output
    y = keras.layers.Dense(hidden_layers[-1], activation='relu')(encoder_output)
    for index in range(len(hidden_layers)-2, -2, -1):
        if index == -1:
          y = keras.layers.Dense(input_dims, activation='sigmoid')(y)
        else:
           y = keras.layers.Dense(hidden_layers[index], activation='relu')(y)

    decoder = keras.models.Model(inputs=encoder_output, outputs=y)
    autoencoder = keras.models.Model(inputs=encoder.input, outputs=y)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder