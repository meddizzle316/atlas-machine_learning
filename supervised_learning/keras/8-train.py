#!/usr/bin/env python3
"""trains model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, save_best=False, filepath=None, verbose=True, shuffle=False):
    """trains model"""
    
    def schedule(epoch, learningRate):
        """learning scheduler"""
        return (alpha / (1 + decay_rate * epoch))
    
    callback_list = []
    if learning_rate_decay:
        learning_rate_sch = K.callbacks.LearningRateScheduler(schedule, 1)
        callback_list.append(learning_rate_sch)

    if early_stopping:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callback_list.append(early_stop)

    # if not filepath:
    filepath = './network1.keras'
    
    if save_best:
        model_checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            mode='min',
        )
        callback_list.append(model_checkpoint)
    
    print(f"This is the callback list {callback_list}")
    print(f"this is the filepath {filepath}")
    if not validation_data:
        r = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    elif validation_data:
        r = network.fit(data, labels, validation_data=validation_data, callbacks=callback_list, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)

    # if not validation_data:
    #     r = network.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    # elif validation_data and not early_stopping and not learning_rate_decay:
    #     r = network.fit(data, labels, validation_data=validation_data, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    # elif validation_data and early_stopping and not learning_rate_decay:
    #     r = network.fit(data, labels, validation_data=validation_data, callbacks=[early_stop], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    # elif validation_data and early_stopping and learning_rate_decay:
    #     r = network.fit(data, labels, validation_data=validation_data, callbacks=[early_stop, learning_rate_sch], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    # elif validation_data and not early_stopping and learning_rate_decay:
    #     r = network.fit(data, labels, validation_data=validation_data, callbacks=[learning_rate_sch], batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    return r