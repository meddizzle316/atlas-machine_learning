#!/usr/bin/env python3
"""creates a convolutional autoencoder model"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder:

    returns encoder, decoder and auto"""

    encoder_input = keras.layers.Input(shape=(input_dims))
    x = keras.layers.Conv2D(filters=filters[0], kernel_size=3, padding='same',
                            activation='relu')(encoder_input)

    for index in range(1, len(filters)):
        x = keras.layers.Conv2D(filters=filters[index], kernel_size=3, padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Conv2D(filters=latent_dims[2], kernel_size=3, padding='same',
                                activation='relu')(x)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)


    encoder = keras.models.Model(encoder_input, x)
    print(encoder.summary())

    encoder_output = encoder.output
    y = keras.layers.Conv2D(filters=latent_dims[2],
                            kernel_size=3,
                            padding='same',
                            activation='relu')(encoder_output)
    y = keras.layers.UpSampling2D((2, 2))(y)
    for index in range(len(filters) - 2, -2, -1):
        if index == -1:
            y = keras.layers.Conv2D(filters=input_dims[2],
                                    kernel_size=3,
                                    padding='same',
                                    activation='sigmoid')(y)
        elif index == 0:
            y = keras.layers.Conv2D(filters=filters[index],
                                    kernel_size=3,
                                    padding='valid',
                                    activation='relu')(y)
            y = keras.layers.UpSampling2D((2, 2))(y)
        else:
            y = keras.layers.Conv2D(filters=filters[index],
                                    kernel_size=3,
                                    padding='same',
                                    activation='relu')(y)
            y = keras.layers.UpSampling2D((2, 2))(y)

    decoder = keras.models.Model(encoder_output, y)
    print(decoder.summary())

    autoencoder = keras.models.Model(encoder_input, y)
    print(autoencoder.summary())

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder