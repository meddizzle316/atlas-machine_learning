#!/usr/bin/env python3
"""trains model"""
import tensorflow.keras as K




def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """trains model"""
    
    def schedule(epoch, learningRate):
        """learning scheduler"""
        return (alpha / (1 + decay_rate * epoch))
    
    learning_rate_sch = K.callbacks.LearningRateScheduler(schedule, 1)

    early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    
    if not validation_data:
        r = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data and not early_stopping and not learning_rate_decay:
        r = network.fit(data, labels, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data and early_stopping and not learning_rate_decay:
        r = network.fit(data, labels, validation_data=validation_data, callbacks=[early_stop], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data and early_stopping and learning_rate_decay:
        r = network.fit(data, labels, validation_data=validation_data, callbacks=[early_stop, learning_rate_sch], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data and not early_stopping and learning_rate_decay:
        r = network.fit(data, labels, validation_data=validation_data, callbacks=[learning_rate_sch], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    return r
