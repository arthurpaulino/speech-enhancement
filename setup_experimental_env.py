from glob import glob
import json
import os

import soundfile as sf
import librosa as lr

from parameters import *
from utils import *


experiment_folder = "data/generated/" + EXPERIMENT_NAME + "/"
experiment_clean_folder = experiment_folder + "clean/"
experiment_noisy_folder = experiment_folder + "noisy/"


for folder in [experiment_clean_folder, experiment_noisy_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

noisy_to_clean_paths = {}

for clean_path in glob(CLEAN_AUDIO_FOLDER + "*"):
    if not is_valid_audio_file(clean_path):
        continue
    y_clean, _ = lr.load(clean_path, sr=SAMPLING_RATE)
    clean_name = filename_from_path(clean_path)

    clean_copy_path = experiment_clean_folder + clean_name + ".wav"

    sf.write(clean_copy_path, y_clean, SAMPLING_RATE)

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

            generated_noisy_file_path = experiment_noisy_folder + filename

            noisy_to_clean_paths[generated_noisy_file_path] = clean_copy_path

            sf.write(generated_noisy_file_path, y_mixed, SAMPLING_RATE)

json.dump(
    noisy_to_clean_paths,
    open(experiment_folder + "noisy_to_clean_pairs.json", "w")
)
