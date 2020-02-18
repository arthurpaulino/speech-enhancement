import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout

from parameters import *


callbacks = [EarlyStopping(monitor='val_loss', patience=2)]


def validate(X_train, Y_train, X_valid, Y_valid):
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    n_layers = len(LAYERS)
    nn = Sequential()

    for layer_info, i in zip(LAYERS, range(n_layers)):
        multiplier, dropout, activation = layer_info
        units = round(multiplier * input_dim)

        if i == 0:
            nn.add(Dense(units, activation=activation, input_dim=input_dim))
        else:
            nn.add(Dense(units, activation=activation))

        nn.add(Dropout(dropout))

    nn.add(Dense(output_dim, activation="sigmoid"))

    nn.compile(optimizer='rmsprop', loss='mse')

    nn.fit(X_train,
           Y_train,
           callbacks=callbacks,
           epochs=100000,
           validation_data=(X_valid, Y_valid))
