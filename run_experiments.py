import json
import os

from sklearn.model_selection import KFold
from numpy.random import seed

from neural_networks import *
from parameters import *
from utils import *

validate_pesq()
seed(RANDOM_SEED)

efm = EXPERIMENT_FOLDER_MAPS
efc = EXPERIMENT_FOLDER + "cleaned/"

if not os.path.exists(efc):
    os.makedirs(efc)

clean_to_noisy = json_load(efm + "clean_to_noisy.json")
noisy_to_clean = json_load(efm + "noisy_to_clean.json")
audio_to_abslt = json_load(efm + "audio_to_abslt.json")
audio_to_abslt_eng = json_load(efm + "audio_to_abslt_eng.json")
audio_to_angle = json_load(efm + "audio_to_angle.json")


def build_X_Y_wrapper(clean_list):
    return build_X_Y(clean_list, clean_to_noisy,
                     audio_to_abslt, audio_to_abslt_eng)


def extract_ys_wrapper(Y_model, lengths, clean_list):
    return extract_ys(Y_model, lengths, clean_list,
                      clean_to_noisy, audio_to_angle)


clean_list = list(clean_to_noisy)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
splits = kf.split(clean_list)

for (train_indexes, valid_indexes), i_fold in zip(splits, range(N_FOLDS)):
    print("############# FOLD {}/{} ##############".format(i_fold + 1, N_FOLDS))
    train_clean = [clean_list[i] for i in train_indexes]
    valid_clean = [clean_list[i] for i in valid_indexes]

    split = round((1 - VALIDATION_RATIO) * len(train_clean))

    train_clean_t = train_clean[:split]
    train_clean_v = train_clean[split:]

    X_train_t, Y_train_t, _ = build_X_Y_wrapper(train_clean_t)
    X_train_v, Y_train_v, _ = build_X_Y_wrapper(train_clean_v)
    X_valid, Y_valid, valid_lengths = build_X_Y_wrapper(valid_clean)

    Y_model = train_and_predict(X_train_t, Y_train_t,
                                X_train_v, Y_train_v,
                                X_valid, Y_valid,
                                i_fold)

    ys_model = extract_ys_wrapper(Y_model, valid_lengths, valid_clean)

    for noisy in ys_model:
        y_noisy = file_to_y(noisy)
        y_clean = file_to_y(noisy_to_clean[noisy])
        y_cleaned = ys_model[noisy]
        filename = filename_from_path(noisy)
        print(filename)
        print(" ", pesq(y_clean, y_noisy), "-->", pesq(y_clean, y_cleaned))
        y_to_file(y_cleaned, efc + filename + ".wav")
