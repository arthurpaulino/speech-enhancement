from gc import collect
from glob import glob
import json
import os

from pypesq import pesq as pesq_fn
import soundfile as sf
import librosa as lr
import numpy as np

from parameters import *


VALID_AUDIO_EXTENSIONS = ["mp3", "ogg", "wav", "flac", "aac", "wma"]

EXPERIMENT_FOLDER = "data/generated/" + EXPERIMENT_NAME + "/"
EXPERIMENT_CLEAN_FOLDER = EXPERIMENT_FOLDER + "clean/"
EXPERIMENT_NOISY_FOLDER = EXPERIMENT_FOLDER + "noisy/"


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


def setup_experiment_env():
    if not os.path.exists(EXPERIMENT_FOLDER):
        os.makedirs(EXPERIMENT_FOLDER)


def generate_audio_files():
    for folder in [EXPERIMENT_CLEAN_FOLDER, EXPERIMENT_NOISY_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    noisy_to_clean_paths = {}

    for clean_path in glob(CLEAN_AUDIO_FOLDER + "*"):
        if not is_valid_audio_file(clean_path):
            continue
        y_clean, _ = lr.load(clean_path, sr=SAMPLING_RATE)
        clean_name = filename_from_path(clean_path)

        sf.write(
            EXPERIMENT_CLEAN_FOLDER + clean_name + ".wav",
            y_clean,
            SAMPLING_RATE
        )

        for noise_path in glob(NOISES_FOLDER + "*"):
            if not is_valid_audio_file(noise_path):
                continue
            y_noise, _ = lr.load(noise_path, sr=SAMPLING_RATE)
            noise_name = filename_from_path(noise_path)

            for noise_db_multiplier in NOISE_DB_MULTIPLIERS:
                y_noise_mult = noise_db_multiplier * y_noise

                y_mixed = filled_sum(
                    y_clean,
                    y_noise_mult[0:min(y_clean.shape[0], y_noise_mult.shape[0])]
                )

                filename = "+".join([
                    clean_name,
                    noise_name,
                    str(round(100 * noise_db_multiplier))
                ]) + ".wav"

                generated_noisy_file_path = EXPERIMENT_NOISY_FOLDER + filename

                noisy_to_clean_paths[generated_noisy_file_path] = clean_path

                sf.write(generated_noisy_file_path, y_mixed, SAMPLING_RATE)

    json.dump(
        noisy_to_clean_paths,
        open(EXPERIMENT_FOLDER + "noisy_to_clean_pairs.json", "w")
    )

    return noisy_to_clean_paths
