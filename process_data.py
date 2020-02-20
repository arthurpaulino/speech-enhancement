from glob import glob
import os

from progress.bar import Bar

from parameters import *
from utils import *


for folder in [EXPERIMENT_FOLDER_CLEAN, EXPERIMENT_FOLDER_NOISY,
               EXPERIMENT_FOLDER_ABSLT, EXPERIMENT_FOLDER_ANGLE,
               EXPERIMENT_FOLDER_ABSLT_ENG, EXPERIMENT_FOLDER_MAPS]:
    if not os.path.exists(folder):
        os.makedirs(folder)

noisy_to_clean = {}
clean_to_noisy = {}
audio_to_abslt = {}
audio_to_angle = {}
audio_to_abslt_eng = {}

clean_path_list = glob(CLEAN_AUDIO_FOLDER_SLASH + "*")
noisy_path_list = glob(NOISES_FOLDER_SLASH + "*")

n_clean = len(clean_path_list)
n_noisy = len(noisy_path_list)
n_noises = len(NOISE_DB_MULTIPLIERS)

bar = Bar("Progress", max=n_clean * n_noisy * n_noises)

for clean_path in clean_path_list:
    if not is_valid_audio_file(clean_path):
        continue
    y_clean = file_to_y(clean_path)
    clean_name = filename_from_path(clean_path)

    clean_copy_path = EXPERIMENT_FOLDER_CLEAN + clean_name + ".wav"

    abslt_path = EXPERIMENT_FOLDER_ABSLT + clean_name + ".pkl"
    angle_path = EXPERIMENT_FOLDER_ANGLE + clean_name + ".pkl"

    clean_to_noisy[clean_copy_path] = []

    abslt, angle = y_to_abslt_angle(y_clean)

    audio_to_abslt[clean_copy_path] = abslt_path
    audio_to_angle[clean_copy_path] = angle_path

    pkl_dump(abslt, abslt_path)
    pkl_dump(angle, angle_path)

    y_to_file(y_clean, clean_copy_path)

    for noise_path in noisy_path_list:
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

            abslt_path = EXPERIMENT_FOLDER_ABSLT + noisy_name + ".pkl"
            angle_path = EXPERIMENT_FOLDER_ANGLE + noisy_name + ".pkl"

            abslt_eng_path = EXPERIMENT_FOLDER_ABSLT_ENG + noisy_name + ".pkl"

            generated_noisy_file_path = EXPERIMENT_FOLDER_NOISY +\
                noisy_name + ".wav"

            noisy_to_clean[generated_noisy_file_path] = clean_copy_path
            clean_to_noisy[clean_copy_path].append(generated_noisy_file_path)

            abslt, angle = y_to_abslt_angle(y_mixed)
            abslt_eng = eng_abslt(abslt)

            audio_to_abslt[generated_noisy_file_path] = abslt_path
            audio_to_angle[generated_noisy_file_path] = angle_path

            audio_to_abslt_eng[generated_noisy_file_path] = abslt_eng_path

            pkl_dump(abslt, abslt_path)
            pkl_dump(angle, angle_path)

            pkl_dump(abslt_eng, abslt_eng_path)

            y_to_file(y_mixed, generated_noisy_file_path)

            bar.next()

bar.finish()

objs = [noisy_to_clean, clean_to_noisy,
        audio_to_abslt, audio_to_angle,
        audio_to_abslt_eng]

filenames = ["noisy_to_clean", "clean_to_noisy",
             "audio_to_abslt", "audio_to_angle",
             "audio_to_abslt_eng"]

for obj, filename in zip(objs, filenames):
    json_dump(obj, EXPERIMENT_FOLDER_MAPS + filename + ".json")
