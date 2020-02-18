import pickle
import json

import pandas as pd
import numpy as np

import soundfile as sf
import librosa as lr

from parameters import *


VALID_AUDIO_EXTENSIONS = ["mp3", "ogg", "wav", "flac", "aac", "wma"]

CLEAN_AUDIO_FOLDER_SLASH = CLEAN_AUDIO_FOLDER + "/"\
    if not CLEAN_AUDIO_FOLDER.endswith("/") else CLEAN_AUDIO_FOLDER

NOISES_FOLDER_SLASH = NOISES_FOLDER + "/"\
    if not NOISES_FOLDER.endswith("/") else NOISES_FOLDER

EXPERIMENT_FOLDER = "data/generated/" + EXPERIMENT_NAME + "/"

EXPERIMENT_FOLDER_MAPS = EXPERIMENT_FOLDER + "maps/"

EXPERIMENT_FOLDER_CLEAN = EXPERIMENT_FOLDER + "clean/"
EXPERIMENT_FOLDER_NOISY = EXPERIMENT_FOLDER + "noisy/"

EXPERIMENT_FOLDER_ABSLT = EXPERIMENT_FOLDER + "abslt/"
EXPERIMENT_FOLDER_ANGLE = EXPERIMENT_FOLDER + "angle/"

EXPERIMENT_FOLDER_ABSLT_ENG = EXPERIMENT_FOLDER + "abslt_eng/"


_n_fft = round(SAMPLING_RATE * FFT_MS / 1000)
_hop_length = round(_n_fft * (1 - OVERLAP))


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


def filled_sum(y_1, y_2):
    if y_1.shape[0] <= y_2.shape[0]:
        sml = y_1
        big = y_2
    else:
        sml = y_2
        big = y_1
    big_size, sml_size = big.shape[0], sml.shape[0]
    s = big.copy()
    for i in range(big_size // sml_size):
        s[i * sml_size : (i + 1) * sml_size] += sml
    mod = big_size % sml_size
    if mod != 0:
        s[big_size - mod : big_size] += sml[0 : mod]
    return s


def y_to_abslt_angle(y):
    D = lr.core.stft(y=y, n_fft=_n_fft, hop_length=_hop_length).T
    return np.abs(D), np.angle(D)


def abslt_angle_to_y(abslt, angle):
    D = abslt * (np.cos(angle) + np.sin(angle) * 1j)
    return lr.core.istft(D.T, hop_length=_hop_length)


def eng_abslt(abslt):
    abslt_df = pd.DataFrame(abslt)

    after_df = pd.concat(
        [abslt_df.shift(-1 * (i + 1)) for i in range(LOOK_AFTER)],
        axis=1
    )

    back_df = pd.concat(
        [abslt_df.shift(i + 1) for i in range(LOOK_BACK)],
        axis=1
    )

    abslt_eng = pd.concat([abslt_df, after_df, back_df], axis=1)

    return abslt_eng.fillna(0).values


def build_X_Y(clean_list, clean_to_noisy, audio_to_abslt, audio_to_abslt_eng):
    X, Y = None, None
    for clean in clean_list:
        clean_matrix = pkl_load(audio_to_abslt[clean])
        for noisy in clean_to_noisy[clean]:
            noisy_matrix = pkl_load(audio_to_abslt_eng[noisy])
            if X is None:
                X = noisy_matrix
                Y = clean_matrix
            else:
                X = np.concatenate([X, noisy_matrix], axis=0)
                Y = np.concatenate([Y, clean_matrix], axis=0)
    return X, Y
