import numpy as np

from keras.utils import get_custom_objects


def get_initial_weights(output_size):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def register_keras_custom_object(cls):
    """ A decorator to register custom layers, loss functions etc in global scope """
    get_custom_objects()[cls.__name__] = cls
    return cls
