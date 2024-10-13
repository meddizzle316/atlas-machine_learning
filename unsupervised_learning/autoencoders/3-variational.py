#!/usr/bin/env python3
"""creates a convolutional autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """makes variational autoencoder"""

    encoder_input = keras.layers.Input(shape=(input_dims,))
    x = keras.layers.Dense(hidden_layers[0], activation='relu')(encoder_input)
    for i in range(1, len(hidden_layers)):
        x = keras.layers.Dense(hidden_layers[i], activation='relu')(x)

    z_mean = keras.layers.Dense(
        latent_dims,
        activation=None,
        name='z_mean')(x)  # had 'relu' instead of none
    z_log_var = keras.layers.Dense(
        latent_dims,
        activation=None,
        name='z_log')(x)  # 'relu instead of none

    def sampling(args):

        z_mean, z_log_var = args
        K = keras.backend

        # epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dims),
        # mean=0., stddev=.1) # might be 1?
        epsilon = K.random_normal(
            shape=(
                K.shape(z_mean)[0],
                K.shape(z_mean)[1]))
        return z_mean + K.exp(z_log_var / 2) * \
            epsilon  # might be z_log_var /2 ??

    z = keras.layers.Lambda(sampling)(
        [z_mean, z_log_var])  # mgiht have to redo?

    encoder = keras.models.Model(
        encoder_input, [
            z, z_mean, z_log_var], name='encoder')
    print(encoder.summary())

    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    y = keras.layers.Dense(hidden_layers[-1], activation='relu')(latent_inputs)

    for i in range(len(hidden_layers) - 2, -1, -1):
        y = keras.layers.Dense(hidden_layers[i], activation='relu')(y)
    y = keras.layers.Dense(input_dims, activation='sigmoid')(y)

    decoder = keras.models.Model(latent_inputs, y, name='decoder')
    print(decoder.summary())

    outputs = decoder(encoder(encoder_input)[0])  # was two last time?
    vae = keras.models.Model(encoder_input, outputs, name='vae')

    vae.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, vae
