#!/usr/bin/env python3
"""creates confusion matrix without dependencies"""
import numpy as np


def create_confusion_matrix(true, pred):
    """confusion matrix"""
    # print(f"this is true's shape {true.shape}")
    # print(f"this is pred's shape {pred.shape}")
    decode_true = np.argmax(true, axis=1)
    # print(f"this is decode true {decode_true} and shape {decode_true.shape}")
    K = len(np.unique(decode_true))
    result = np.zeros((K, K))
    # print(f"this is the initial result {result} and shape {result.shape}")
    decode_pred = np.argmax(pred, axis=1)
    # print(f"this is decode pred {decode_pred} and shape {decode_pred.shape}")
    for i in range(len(decode_true)):
        result[decode_true[i]][decode_pred[i]] += 1

    return result
