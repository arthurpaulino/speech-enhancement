from glob import glob
import os

from progress.bar import Bar

from parameters import *
from utils import *


# creating the folder structure for the experiment
for folder in [EXPERIMENT_FOLDER_CLEAN, EXPERIMENT_FOLDER_NOISY_EXP,
               EXPERIMENT_FOLDER_MAGNI, EXPERIMENT_FOLDER_PHASE,
               EXPERIMENT_FOLDER_MAGNI_ENG, EXPERIMENT_FOLDER_MAPS]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# maps of relationships for later data retrieval
noisy_to_clean = {}
clean_to_noisy = {}
audio_to_magni = {}
audio_to_phase = {}
audio_to_magni_eng = {}

# gathering paths for clean and noisy files
clean_path_list = glob(CLEAN_AUDIO_FOLDER_SLASH + "*")
noisy_path_list = glob(NOISES_FOLDER_SLASH + "*")

n_clean = len(clean_path_list)
n_noisy = len(noisy_path_list)
n_noises = len(SNRS)

bar = Bar("Progress", max=n_clean * n_noisy * n_noises)

for clean_path in clean_path_list:
    if not is_valid_audio_file(clean_path):
        continue
    y_clean = file_to_y(clean_path)
    clean_name = filename_from_path(clean_path)

    clean_copy_path = EXPERIMENT_FOLDER_CLEAN + clean_name + ".wav"

    magni_path = EXPERIMENT_FOLDER_MAGNI + clean_name + ".pkl"
    phase_path = EXPERIMENT_FOLDER_PHASE + clean_name + ".pkl"

    clean_to_noisy[clean_copy_path] = []

    magni, phase = y_to_magni_phase(y_clean)

    audio_to_magni[clean_copy_path] = magni_path
    audio_to_phase[clean_copy_path] = phase_path

    pkl_dump(magni, magni_path)
    pkl_dump(phase, phase_path)

    y_to_file(y_clean, clean_copy_path)

    for noise_path in noisy_path_list:
        if not is_valid_audio_file(noise_path):
            continue
        y_noise = file_to_y(noise_path)
        noise_name = filename_from_path(noise_path)

        for snr in SNRS:
            multiplier = noise_multiplier(y_clean, y_noise, snr)
            y_noise_mult = multiplier * y_noise

            y_noise_mult_extended = extend(y_noise_mult, y_clean.shape[0])

            y_mixed = y_clean + y_noise_mult_extended

            noisy_name = "|".join([clean_name, noise_name, str(snr)])

            magni_path = EXPERIMENT_FOLDER_MAGNI + noisy_name + ".pkl"
            phase_path = EXPERIMENT_FOLDER_PHASE + noisy_name + ".pkl"

            magni_eng_path = EXPERIMENT_FOLDER_MAGNI_ENG + noisy_name + ".pkl"

            generated_noisy_file_path = EXPERIMENT_FOLDER_NOISY_EXP +\
                noisy_name + ".wav"

            noisy_to_clean[generated_noisy_file_path] = clean_copy_path
            clean_to_noisy[clean_copy_path].append(generated_noisy_file_path)

            magni, phase = y_to_magni_phase(y_mixed)
            magni_eng = eng_magni(magni)

            audio_to_magni[generated_noisy_file_path] = magni_path
            audio_to_phase[generated_noisy_file_path] = phase_path

            audio_to_magni_eng[generated_noisy_file_path] = magni_eng_path

            pkl_dump(magni, magni_path)
            pkl_dump(phase, phase_path)

            pkl_dump(magni_eng, magni_eng_path)

            y_to_file(y_mixed, generated_noisy_file_path)

            bar.next()

bar.finish()

objs = [noisy_to_clean, clean_to_noisy,
        audio_to_magni, audio_to_phase,
        audio_to_magni_eng]

filenames = ["noisy_to_clean", "clean_to_noisy",
             "audio_to_magni", "audio_to_phase",
             "audio_to_magni_eng"]

for obj, filename in zip(objs, filenames):
    json_dump(obj, EXPERIMENT_FOLDER_MAPS + filename + ".json")
