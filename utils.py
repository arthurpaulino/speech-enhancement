import pickle
import json

import numpy as np

import soundfile as sf
import librosa as lr

from parameters import *


CLEAN_AUDIO_FOLDER_SLASH = CLEAN_AUDIO_FOLDER + "/"\
    if not CLEAN_AUDIO_FOLDER.endswith("/") else CLEAN_AUDIO_FOLDER

NOISES_FOLDER_SLASH = NOISES_FOLDER + "/"\
    if not NOISES_FOLDER.endswith("/") else NOISES_FOLDER

EXPERIMENT_FOLDER = "data/generated/" + EXPERIMENT_NAME + "/"

EXPERIMENT_FOLDER_CLEAN = EXPERIMENT_FOLDER + "clean/"
EXPERIMENT_FOLDER_NOISY = EXPERIMENT_FOLDER + "noisy/"

EXPERIMENT_FOLDER_ABSLT = EXPERIMENT_FOLDER + "abslt/"
EXPERIMENT_FOLDER_ANGLE = EXPERIMENT_FOLDER + "angle/"


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


def json_load(path):
    return json.load(open(path, "r"))


def json_dump(obj, path):
    json.dump(obj, open(path, "w"))


def pkl_load(path):
    return pickle.load(open(path, "rb"))


def pkl_dump(obj, path):
    pickle.dump(obj, open(path, "wb"))
