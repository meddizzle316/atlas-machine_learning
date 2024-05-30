#!/usr/bin/env python3
"""module for decoding one_hot"""

def one_hot_decode(one_hot):
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
