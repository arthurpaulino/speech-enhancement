from gc import collect
from glob import glob
import os

import soundfile as sf
import librosa as lr

from parameters import *
from utils import *


for folder in [GENERATED_CLEAN_FOLDER, GENERATED_NOISY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for clean_path in glob(CLEAN_AUDIO_FOLDER + "*"):
    if not is_valid_audio_file(clean_path):
        continue
    y_clean, _ = lr.load(clean_path, sr=SAMPLING_RATE)
    clean_name = filename_from_path(clean_path)
    sf.write(
        GENERATED_CLEAN_FOLDER + clean_name + ".wav",
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
                y_noise_mult[0 : min(y_clean.shape[0], y_noise_mult.shape[0])]
            )
            filename = "+".join(
                [
                    clean_name,
                    noise_name,
                    str(round(100 * noise_db_multiplier))
                ]
            ) + ".wav"
            sf.write(GENERATED_NOISY_FOLDER + filename, y_mixed, SAMPLING_RATE)
            exit()
