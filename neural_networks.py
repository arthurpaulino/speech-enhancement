import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from numpy.random import shuffle

from sklearn.preprocessing import MinMaxScaler

from parameters import *
from utils import *


early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE)


def add_hidden_layers(nn, input_dim, output_dim):

    for layer_info, i in zip(LAYERS, range(len(LAYERS))):
        multiplier, activation, dropout = layer_info
        units = round(multiplier * input_dim)

        nn.add(Dense(units, activation=activation, input_shape=(input_dim,)))
        nn.add(Dropout(dropout))


def add_hidden_layers_(nn, input_dim, output_dim):

    n_rows = LOOK_BACK + 1 + LOOK_AFTER
    n_cols = output_dim

    nn.add(Reshape((n_rows, n_cols, 1), input_shape=(input_dim,)))
    nn.add(Conv2D(64,
           kernel_size=(3, 3),
           activation="relu"))
    nn.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    nn.add(Dropout(0.25))
    nn.add(Flatten())


def build_dense_tensor(input_dim, output_dim):
    t = Sequential()
    t.add(Dense(input_dim, activation="relu", input_shape=(input_dim,)))
    t.add(Dropout(0.25))
    return t


def build_conv_tensor(input_dim, output_dim):
    n_rows = LOOK_BACK + 1 + LOOK_AFTER
    n_cols = output_dim

    t = Sequential()
    t.add(Reshape((n_rows, n_cols, 1), input_shape=(input_dim,)))
    t.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    t.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    t.add(Dropout(0.25))
    t.add(Flatten())
    return t


def build_nn(input_dim, output_dim):
    nn = Sequential()

    add_hidden_layers_(nn, input_dim, output_dim)

    nn.add(Dense(output_dim, activation="sigmoid"))

    nn.compile(optimizer="rmsprop", loss="mae")

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
