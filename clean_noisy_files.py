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

    y_to_file(y_noisy, EXPERIMENT_FOLDER_NOISY + noisy_name + ".wav")

    ampli, phase = y_to_ampli_phase(y_noisy)

    ampli_eng = eng_ampli(ampli)

    Ys_models = []

    for model_path in model_path_list:
        model = pkl_load(model_path)
        Ys_models.append(model.predict(ampli_eng))

    Y_models = ensemble(Ys_models)

    y_cleaned = ampli_phase_to_y(Y_models, phase)

    y_to_file(y_cleaned, EXPERIMENT_FOLDER_CLEANED + noisy_name + ".wav")

    bar.next()

bar.finish()
