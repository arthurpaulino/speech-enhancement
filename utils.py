from pypesq import pesq as pesq_fn
import librosa as lr

from parameters import *


def is_valid_audio_file(path):
    extension = path.split(".")[-1]
    return extension in VALID_AUDIO_EXTENSIONS


def filename_from_path(path):
    return path.split("/")[-1].split(".")[0]


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


def pesq(y_ref, y_deg):
    if PESQ_SAMPLING_RATE != SAMPLING_RATE:
        y_ref = lr.core.resample(y_ref, SAMPLING_RATE, PESQ_SAMPLING_RATE)
        y_deg = lr.core.resample(y_deg, SAMPLING_RATE, PESQ_SAMPLING_RATE)
    return pesq_fn(y_ref, y_deg, PESQ_SAMPLING_RATE)
