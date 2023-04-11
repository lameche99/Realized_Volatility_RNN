import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RNN:

    def __init__(self, x_train: np.array, y_train: np.array, layers: int):
        self.x_train = x_train
        self.y_train = y_train
        self.layers = layers

    