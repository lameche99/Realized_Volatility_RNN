from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, GRU
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def get_XY(data: pd.DataFrame, xlabs: list, ylab: list, scaler: MinMaxScaler, time_steps: int = 2):
    """
    This function receives a dataframe, feature and target labels as well
    the number of time steps for each sequence and returns a 3D and a 2D scaled arrays
    for RNN
    :param data: pd.DataFrame - dataset of features and target variables
    :param xlabs: list - feature labels
    :param ylab: list - target label
    :param time_steps: int - length of each input sequence, default = 2
    :return: tuple(np.array, np.array) - X-shape: (observations, time_steps, features)
    Y-shape: (observations,)
    """
    # extract input sequences and output
    # cpy = data.copy()
    # cpy[xlabs] = scaler.fit_transform(data[xlabs])
    # cpy[ylab] = scaler.fit_transform(data[ylab])

    X, Y = [], []
    for i in range(time_steps, data.shape[0]):
        X.append(data[xlabs].iloc[i - time_steps : i])
        Y.append(data[ylab].iloc[i])
    
    X, Y = np.array(X), np.array(Y).reshape((len(Y),))
    return X, Y


class RNN:

    def __init__(self, x_train: np.array, y_train: np.array, units: int,
                    type_: str = 'lstm', epochs: int = 100, batch_size: int = 32,
                        activation: str = 'tanh'):
        self.x_train = x_train
        self.y_train = y_train
        self.units = units
        self.regressor = self.create_and_fit_NN(
            type_,
            epochs,
            batch_size,
            activation
        )

    def create_and_fit_NN(self, type_, epochs: int, batch_size: int, activation: str):
        """
        This function creates a Recurrent Neural Network, compiles it and fits the
        object's x_train and y_train datasets.
        :param simple: bool - whether we are building an Elman or LSTM network, default = False
        :param epochs: int - number of epochs for the NN, default = 100
        :param batch_size: int - batch size for NN, default = 32
        :param activation: str - activation function for hidden and output layers, default = 'relu'
        :return: Sequential - Sequential object for a fitted RNN
        """

        # initialize sequential RNN
        mod = Sequential()
        # add first layer with x_train inputs of shape: (time_step x features)
        if type_ == 'simple':
            mod.add(SimpleRNN(
                units=self.units,
                activation=activation,
                # return_sequences=True,
                input_shape=(self.x_train.shape[1], self.x_train.shape[2])
            ))
        elif type_ == 'lstm':
            mod.add(LSTM(
                units=self.units,
                activation=activation,
                # return_sequences=True,
                input_shape=(self.x_train.shape[1], self.x_train.shape[2])
            ))
        elif type_ == 'gru':
            mod.add(GRU(
                units=self.units,
                activation=activation,
                # return_sequences=True,
                input_shape=(self.x_train.shape[1], self.x_train.shape[2])
            ))
        else:
            raise Exception('Incorrect Network type. Acceptable inputs: "simple", "lstm", "gru".')
        # add output layer
        mod.add(Dense(
            units=1,
            activation='tanh'
        ))
        # compile RNN
        mod.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        # Fit X_train and Y_train to the RNN
        mod.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        
        return mod
