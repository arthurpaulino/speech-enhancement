from glob import glob
import shutil
import os

from progress.bar import Bar

from utils import *

for folder in EXPERIMENT_FOLDER_NOISY, EXPERIMENT_FOLDER_CLEANED:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

noisy_path_list = glob(NOISY_FOLDER_SLASH + "*")
model_path_list = glob(EXPERIMENT_FOLDER_MODELS + "*")

bar = Bar("Progress", max=len(noisy_path_list))

for noisy_path in noisy_path_list:
    if not is_valid_audio_file(noisy_path):
        continue
    y_noisy = file_to_y(noisy_path)
    noisy_name = filename_from_path(noisy_path)

    noisy_copy_path = EXPERIMENT_FOLDER_NOISY + noisy_name + ".wav"

    y_to_file(y_noisy, noisy_copy_path)

    abslt, angle = y_to_abslt_angle(y)

    # todo: iterate on models

    bar.next()

bar.finish()
