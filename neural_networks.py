import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from numpy.random import shuffle

from sklearn.preprocessing import MinMaxScaler

from parameters import *
from utils import *


early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)


def build_nn(input_dim, output_dim):
    n_layers = len(LAYERS)
    nn = Sequential()

    for layer_info, i in zip(LAYERS, range(n_layers)):
        multiplier, activation, dropout = layer_info
        units = round(multiplier * input_dim)

        if i == 0:
            nn.add(Dense(units, activation=activation, input_dim=input_dim))
        else:
            nn.add(Dense(units, activation=activation))

        nn.add(Dropout(dropout))

    nn.add(Dense(output_dim, activation="sigmoid"))

    nn.compile(optimizer="rmsprop", loss="mse")

    return nn


def train_and_predict(X_train_t, Y_train_t,
                      X_train_v, Y_train_v,
                      X_valid, Y_valid,
                      seed, model_id):
    input_len, input_dim = X_train_t.shape
    output_dim = Y_train_t.shape[1]

    X_scaler, Y_scaler = MinMaxScaler(), MinMaxScaler()

    X_train_t_scaled = X_scaler.fit_transform(X_train_t)
    Y_train_t_scaled = Y_scaler.fit_transform(Y_train_t)

    X_train_v_scaled = X_scaler.transform(X_train_v)
    Y_train_v_scaled = Y_scaler.transform(Y_train_v)

    X_valid_scaled = X_scaler.transform(X_valid)
    Y_valid_scaled = Y_scaler.transform(Y_valid)

    nn = build_nn(input_dim, output_dim)

    callbacks = [
        early_stop,
        ModelCheckpoint(
            filepath=EXPERIMENT_FOLDER + "models/" + model_id + ".h5",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    nn.fit(X_train_t_scaled,
           Y_train_t_scaled,
           batch_size=round(BATCH_SIZE_RATIO * input_len),
           epochs=1_000_000, # going to use early stop instead
           verbose=VERBOSE,
           callbacks=callbacks,
           validation_data=(X_train_v_scaled, Y_train_v_scaled))

    Y_model_scaled = nn.predict(X_valid_scaled)
    return Y_scaler.inverse_transform(Y_model_scaled)
