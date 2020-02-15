from glob import glob
import os

from parameters import *
from utils import *


clean_audio_folder_slash = CLEAN_AUDIO_FOLDER + "/"\
    if not CLEAN_AUDIO_FOLDER.endswith("/") else CLEAN_AUDIO_FOLDER

noises_folder_slash = NOISES_FOLDER + "/"\
    if not NOISES_FOLDER.endswith("/") else NOISES_FOLDER

experiment_folder = "data/generated/" + EXPERIMENT_NAME + "/"

experiment_clean_folder = experiment_folder + "clean/"
experiment_noisy_folder = experiment_folder + "noisy/"

experiment_abslt_folder = experiment_folder + "abslt/"
experiment_angle_folder = experiment_folder + "angle/"

for folder in [experiment_clean_folder, experiment_noisy_folder,
               experiment_abslt_folder, experiment_angle_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

noisy_to_clean = {}
clean_to_noisy = {}
audio_to_abslt = {}
audio_to_angle = {}

for clean_path in glob(clean_audio_folder_slash + "*"):
    if not is_valid_audio_file(clean_path):
        continue
    y_clean = file_to_y(clean_path)
    clean_name = filename_from_path(clean_path)

    clean_copy_path = experiment_clean_folder + clean_name + ".wav"

    abslt_path = experiment_abslt_folder + clean_name + ".pkl"
    angle_path = experiment_angle_folder + clean_name + ".pkl"

    clean_to_noisy[clean_copy_path] = []

    abslt, angle = y_to_abslt_angle(y_clean)

    audio_to_abslt[clean_copy_path] = abslt_path
    audio_to_angle[clean_copy_path] = angle_path

    pkl_dump(abslt, abslt_path)
    pkl_dump(angle, angle_path)

    y_to_file(y_clean, clean_copy_path)

    for noise_path in glob(noises_folder_slash + "*"):
        if not is_valid_audio_file(noise_path):
            continue
        y_noise = file_to_y(noise_path)
        noise_name = filename_from_path(noise_path)

        for noise_db_multiplier in NOISE_DB_MULTIPLIERS:
            y_noise_mult = noise_db_multiplier * y_noise

            y_mixed = filled_sum(
                y_clean,
                y_noise_mult[0:min(y_clean.shape[0], y_noise_mult.shape[0])]
            )

            noisy_name = "+".join([
                clean_name,
                noise_name,
                str(round(100 * noise_db_multiplier))
            ])

            abslt_path = experiment_abslt_folder + noisy_name + ".pkl"
            angle_path = experiment_angle_folder + noisy_name + ".pkl"

            generated_noisy_file_path = experiment_noisy_folder +\
                noisy_name + ".wav"

            noisy_to_clean[generated_noisy_file_path] = clean_copy_path
            clean_to_noisy[clean_copy_path].append(generated_noisy_file_path)

            abslt, angle = y_to_abslt_angle(y_mixed)

            audio_to_abslt[generated_noisy_file_path] = abslt_path
            audio_to_angle[generated_noisy_file_path] = angle_path

            pkl_dump(abslt, abslt_path)
            pkl_dump(angle, angle_path)

            y_to_file(y_mixed, generated_noisy_file_path)

for obj, filename in zip(
        [noisy_to_clean, clean_to_noisy, audio_to_abslt, audio_to_angle],
        ["noisy_to_clean", "clean_to_noisy", "audio_to_abslt", "audio_to_angle"]
    ):

    json_dump(obj, experiment_folder + filename + ".json")
