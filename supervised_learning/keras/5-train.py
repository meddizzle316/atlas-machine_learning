import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, verbose=True, shuffle=False):
    """trains model"""
    if not validation_data:
        r = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    else:
        r = network.fit(data, labels, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    return r
