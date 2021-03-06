from time import time
import pickle
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.models import Model as NN

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import soundfile as sf
import librosa as lr

from pesq import pesq
from pystoi.stoi import stoi

from parameters import *


VALID_AUDIO_EXTENSIONS = ["mp3", "ogg", "wav", "flac", "aac", "wma"]

CLEAN_AUDIO_FOLDER_SLASH = CLEAN_AUDIO_FOLDER + "/"\
    if not CLEAN_AUDIO_FOLDER.endswith("/") else CLEAN_AUDIO_FOLDER

NOISES_FOLDER_SLASH = NOISES_FOLDER + "/"\
    if not NOISES_FOLDER.endswith("/") else NOISES_FOLDER

NOISY_FOLDER_SLASH = NOISY_FOLDER + "/"\
    if not NOISY_FOLDER.endswith("/") else NOISY_FOLDER

EXPERIMENT_FOLDER = "data/experiments/" + EXPERIMENT_NAME + "/"

EXPERIMENT_FOLDER_MAPS = EXPERIMENT_FOLDER + "maps/"

EXPERIMENT_FOLDER_CLEAN = EXPERIMENT_FOLDER + "clean/"
EXPERIMENT_FOLDER_NOISY_EXP = EXPERIMENT_FOLDER + "noisy_exp/"

EXPERIMENT_FOLDER_ABSLT = EXPERIMENT_FOLDER + "abslt/"
EXPERIMENT_FOLDER_ANGLE = EXPERIMENT_FOLDER + "angle/"

EXPERIMENT_FOLDER_ABSLT_ENG = EXPERIMENT_FOLDER + "abslt_eng/"

EXPERIMENT_FOLDER_CLEANED_EXP = EXPERIMENT_FOLDER + "cleaned_exp/"
EXPERIMENT_FOLDER_MODELS = EXPERIMENT_FOLDER + "models/"

EXPERIMENT_FOLDER_NOISY = EXPERIMENT_FOLDER + "noisy/"
EXPERIMENT_FOLDER_CLEANED = EXPERIMENT_FOLDER + "cleaned/"

EXPERIMENT_FOLDER_REVERSE = EXPERIMENT_FOLDER + "reverse/"


_n_fft = round(SAMPLING_RATE * FFT_MS / 1000)
_hop_length = round(_n_fft * (1 - OVERLAP))


############################ FILES


def is_valid_audio_file(path):
    extension = path.split(".")[-1]
    return extension in VALID_AUDIO_EXTENSIONS


def filename_from_path(path):
    return path.split("/")[-1].split(".")[0]


def file_to_y(path):
    y, _ = lr.load(path, sr=SAMPLING_RATE)
    return y


def y_to_file(y, path):
    sf.write(path, y, SAMPLING_RATE)


def json_load(path):
    return json.load(open(path, "r"))


def json_dump(obj, path):
    json.dump(obj, open(path, "w"))


def pkl_load(path):
    return pickle.load(open(path, "rb"))


def pkl_dump(obj, path):
    pickle.dump(obj, open(path, "wb"))


############################ ARRAYS


def extend(y, size):
    y_size = y.shape[0]
    if size <= y_size:
        return y[:size]
    else:
        y_new = np.zeros(size)
        for i in range(size // y_size):
            y_new[i * y_size : (i + 1) * y_size] += y
        mod = size % y_size
        if mod != 0:
            y_new[size - mod : size] += y[0 : mod]
        return y_new


def y_to_abslt_angle(y):
    D = lr.core.stft(y=y, n_fft=_n_fft, hop_length=_hop_length).T
    return np.abs(D), np.angle(D)


def abslt_angle_to_y(abslt, angle):
    D = abslt * (np.cos(angle) + np.sin(angle) * 1j)
    return lr.core.istft(D.T, hop_length=_hop_length)


def eng_abslt(abslt):
    abslt_df = pd.DataFrame(abslt)

    to_concat = []

    if LOOK_BACK > 0:
        back_df = pd.concat(
            [abslt_df.shift(i + 1) for i in range(LOOK_BACK)],
            axis=1
        )
        to_concat.append(back_df)

    to_concat.append(abslt_df)

    if LOOK_AFTER > 0:
        after_df = pd.concat(
            [abslt_df.shift(-1 * (i + 1)) for i in range(LOOK_AFTER)],
            axis=1
        )
        to_concat.append(after_df)

    if LOOK_BACK > 0 or LOOK_AFTER > 0:
        abslt_eng = pd.concat(to_concat, axis=1)
    else:
        abslt_eng = abslt_df

    return abslt_eng.fillna(0).values


def concatenate(Xs, Ys):
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def build_X_Y(clean_list, clean_to_noisy, audio_to_abslt, audio_to_abslt_eng):
    noisy_matrices = []
    clean_matrices = []
    lengths = []
    for clean in clean_list:
        clean_matrix = pkl_load(audio_to_abslt[clean])
        lengths.append(clean_matrix.shape[0])
        for noisy in clean_to_noisy[clean]:
            noisy_matrix = pkl_load(audio_to_abslt_eng[noisy])
            noisy_matrices.append(noisy_matrix)
            clean_matrices.append(clean_matrix)
    X, Y = concatenate(noisy_matrices, clean_matrices)
    return X, Y, lengths


def ensemble(Ys_models):
    M = np.array(Ys_models)
    means = M.mean(axis=0)
    weights = 1 / np.maximum(np.abs(M - means), 1e-10) ** ENSEMBLE_WEIGHTS_POWER
    return np.average(M, weights=weights, axis=0)


def extract_ys(Y_model, lengths, clean_list, clean_to_noisy, audio_to_angle):
    ys_model = {}
    cumulative_length = 0
    for clean, length in zip(clean_list, lengths):
        for noisy in clean_to_noisy[clean]:
            abslt = Y_model[cumulative_length : cumulative_length + length, :]
            angle = pkl_load(audio_to_angle[noisy])
            ys_model[noisy] = abslt_angle_to_y(abslt, angle)
            cumulative_length += length
    return ys_model


def cap(y_1, y_2):
    size = min(y_1.shape[0], y_2.shape[0])
    return y_1[:size], y_2[:size]


############################ AUDIO


def noise_multiplier(y_clean, y_noise, snr):
    return (y_clean.var() / y_noise.var() / (10 ** (snr / 10))) ** 0.5


def validate_pesq():
    to_quit = False
    if PESQ_MODE not in ["nb", "wb"]:
        print("Invalid PESQ mode")
        to_quit = True
    if PESQ_SAMPLING_RATE not in [8000, 16000]:
        print("Invalid PESQ sampling rate")
        to_quit = True
    if PESQ_SAMPLING_RATE == 8000 and PESQ_MODE != "nb":
        print("Invalid PESQ sampling rate for 'nb' mode")
        to_quit = True
    if to_quit:
        exit()


def snr_fn(y_truth, y_valid):
    y_truth, y_valid = cap(y_truth, y_valid)
    return 10 * np.log10(y_truth.var() / (y_truth - y_valid).var())


def pesq_fn(y_truth, y_valid):
    y_truth, y_valid = cap(y_truth, y_valid)
    if SAMPLING_RATE != PESQ_SAMPLING_RATE:
        y_truth = lr.core.resample(y_truth, SAMPLING_RATE, PESQ_SAMPLING_RATE)
        y_valid = lr.core.resample(y_valid, SAMPLING_RATE, PESQ_SAMPLING_RATE)
    return pesq(PESQ_SAMPLING_RATE, y_truth, y_valid, PESQ_MODE)


def stoi_fn(y_truth, y_valid):
    y_truth, y_valid = cap(y_truth, y_valid)
    return stoi(y_truth, y_valid, SAMPLING_RATE, extended=False)


def estoi_fn(y_truth, y_valid):
    y_truth, y_valid = cap(y_truth, y_valid)
    return stoi(y_truth, y_valid, SAMPLING_RATE, extended=True)


############################ NEURAL NETWORKS


def build_tensor(layers, tensor_input):
    tensor = layers[0](tensor_input)
    for layer in layers[1:]:
        tensor = layer(tensor)
    return tensor


def build_nn(input_dim, output_dim):
    nn_input = Input((input_dim,))

    dense_layers = [
        Dense(input_dim, activation="relu"),
        Dropout(0.25)
    ]

    dense_tensor = build_tensor(dense_layers, nn_input)

    conv_layers_1 = [
        Reshape((LOOK_BACK + 1 + LOOK_AFTER, output_dim, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2))
    ]

    conv_layers_2 = [
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2))
    ]

    conv_tensor_1 = build_tensor(conv_layers_1, nn_input)
    conv_tensor_2 = build_tensor(conv_layers_2, conv_tensor_1)

    conv_tensor_1_flat = Flatten()(conv_tensor_1)
    conv_tensor_2_flat = Flatten()(conv_tensor_2)

    concat = Concatenate()([dense_tensor,
                            conv_tensor_1_flat,
                            conv_tensor_2_flat])

    nn_output = Dense(output_dim, activation="sigmoid")(concat)

    nn = NN(nn_input, nn_output)

    nn.compile(optimizer="adam", loss="mae")

    return nn


class Model:
    def __init__(self, X_scaler, Y_scaler, nn):
        self.X_scaler = X_scaler
        self.Y_scaler = Y_scaler
        self.nn = nn

    def predict(self, X):
        return self.Y_scaler.inverse_transform(
            self.nn.predict(self.X_scaler.transform(X))
        )


def train_and_predict(X_train, Y_train, X_predi,
                      X_valid=None, Y_valid=None, save_model=True):
    input_len, input_dim = X_train.shape
    output_dim = Y_train.shape[1]

    X_scaler, Y_scaler = MinMaxScaler(), MinMaxScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    Y_train_scaled = Y_scaler.fit_transform(Y_train)

    nn = build_nn(input_dim, output_dim)

    validation_data = None
    min_delta = MIN_DELTA
    monitor = "loss"

    if X_valid is not None and Y_valid is not None:
        X_valid_scaled = X_scaler.fit_transform(X_valid)
        Y_valid_scaled = Y_scaler.fit_transform(Y_valid)
        validation_data = (X_valid_scaled, Y_valid_scaled)
        min_delta = 0
        monitor = "val_loss"

    callbacks=[EarlyStopping(monitor=monitor,
                             min_delta=min_delta,
                             patience=PATIENCE,
                             restore_best_weights=True)]

    nn.fit(X_train_scaled,
           Y_train_scaled,
           batch_size=round(BATCH_SIZE_RATIO * input_len),
           epochs=1_000_000, # going to use early stop instead
           verbose=VERBOSE,
           callbacks=callbacks,
           validation_data=validation_data)

    model = Model(X_scaler, Y_scaler, nn)

    if save_model:
        pkl_dump(model, EXPERIMENT_FOLDER_MODELS + str(round(time())) + ".pkl")

    return model.predict(X_predi)
