from numpy.random import seed as nseed
from random import shuffle, seed
import json

from neural_networks import *
from parameters import *
from utils import *


seed(RANDOM_SEED)
nseed(RANDOM_SEED)

efm = EXPERIMENT_FOLDER_MAPS

clean_to_noisy = json_load(efm + "clean_to_noisy.json")
audio_to_abslt = json_load(efm + "audio_to_abslt.json")
audio_to_abslt_eng = json_load(efm + "audio_to_abslt_eng.json")
audio_to_angle = json_load(efm + "audio_to_angle.json")

def build_X_Y_wrapper(clean_list):
    return build_X_Y(clean_list, clean_to_noisy,
                     audio_to_abslt, audio_to_abslt_eng)

clean_list = list(clean_to_noisy)

split = round((1 - VALIDATION_RATIO) * len(clean_list))

for i_experiment in range(N_EXPERIMENTS):
    print("############## EXPERIMENT {}".format(i_experiment + 1))
    shuffle(clean_list)
    train_clean = clean_list[:split]
    valid_clean = clean_list[split:]

    X_train, Y_train, train_lengths = build_X_Y_wrapper(train_clean)
    X_valid, Y_valid, valid_lengths = build_X_Y_wrapper(valid_clean)

    Y_model = train_and_predict(X_train, Y_train, X_valid, Y_valid)

    ys = extract_ys(Y_model, valid_lengths, valid_clean,
                    clean_to_noisy, audio_to_angle)
