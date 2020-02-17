from random import shuffle, seed
import json

from parameters import *
from utils import *


seed(RANDOM_SEED)

efm = EXPERIMENT_FOLDER_MAPS

clean_to_noisy = json_load(efm + "clean_to_noisy.json")
audio_to_abslt = json_load(efm + "audio_to_abslt.json")
audio_to_abslt_eng = json_load(efm + "audio_to_abslt_eng.json")

def build_X_Y_wrapper(clean_list):
    return build_X_Y(clean_list, clean_to_noisy,
                     audio_to_abslt, audio_to_abslt_eng)

clean_list = list(clean_to_noisy)

split = round((1 - VALIDATION_RATIO) * len(clean_list))

for i_experiment in range(N_EXPERIMENTS):
    shuffle(clean_list)
    train_clean = clean_list[:split]
    valid_clean = clean_list[split:]

    X_train, Y_train = build_X_Y_wrapper(train_clean)
    X_valid, Y_valid = build_X_Y_wrapper(valid_clean)

    validate(X_train, Y_train, X_valid, Y_valid)
    break
