#!/usr/bin/env python3
"""trains model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, verbose=True, shuffle=False):
    """trains model"""
    
    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    if not validation_data:
        r = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data and not early_stopping:
        r = network.fit(data, labels, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    else:
        r = network.fit(data, labels, validation_data=validation_data, callbacks=[early_stop], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    return r
