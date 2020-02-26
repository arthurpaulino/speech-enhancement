from time import time
import shutil
import json
import os

from sklearn.model_selection import KFold
from numpy.random import seed
import pandas as pd

from neural_networks import *
from parameters import *
from utils import *

validate_pesq()
seed(RANDOM_SEED)

efc = EXPERIMENT_FOLDER + "cleaned/"
efm = EXPERIMENT_FOLDER + "models/"

for folder in efc, efm:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

clean_to_noisy = json_load(EXPERIMENT_FOLDER_MAPS + "clean_to_noisy.json")
noisy_to_clean = json_load(EXPERIMENT_FOLDER_MAPS + "noisy_to_clean.json")
audio_to_abslt = json_load(EXPERIMENT_FOLDER_MAPS + "audio_to_abslt.json")
audio_to_angle = json_load(EXPERIMENT_FOLDER_MAPS + "audio_to_angle.json")
audio_to_abslt_eng = json_load(
    EXPERIMENT_FOLDER_MAPS + "audio_to_abslt_eng.json"
)


def build_X_Y_wrapper(clean_list):
    return build_X_Y(clean_list, clean_to_noisy,
                     audio_to_abslt, audio_to_abslt_eng)


def extract_ys_wrapper(Y_model, lengths, clean_list):
    return extract_ys(Y_model, lengths, clean_list,
                      clean_to_noisy, audio_to_angle)


clean_list = list(clean_to_noisy)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
splits = kf.split(clean_list)

report = []

for (train_indexes, valid_indexes), i_fold in zip(splits, range(N_FOLDS)):
    print("\nFold {}/{}".format(i_fold + 1, N_FOLDS))
    start = time()

    train_clean = [clean_list[i] for i in train_indexes]
    valid_clean = [clean_list[i] for i in valid_indexes]

    X_valid, Y_valid, valid_lengths = build_X_Y_wrapper(valid_clean)

    if 0 < INNER_VALIDATION and INNER_VALIDATION < 1:
        split = round((1 - INNER_VALIDATION) * len(train_clean))

        train_clean_t = train_clean[:split]
        train_clean_v = train_clean[split:]

        X_train_t, Y_train_t, _ = build_X_Y_wrapper(train_clean_t)
        X_train_v, Y_train_v, _ = build_X_Y_wrapper(train_clean_v)

        model_id = "nn_" + str(i_fold)

        Y_model = train_and_predict(X_train_t, Y_train_t,
                                    X_train_v, Y_train_v,
                                    X_valid, Y_valid,
                                    i_fold, model_id)
    elif INNER_VALIDATION > 1 and isinstance(INNER_VALIDATION, int):
        inner_kf = KFold(
            n_splits=INNER_VALIDATION,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        inner_splits = inner_kf.split(train_clean)

        iter = zip(inner_splits, range(INNER_VALIDATION))

        Y_model = None

        Ys_models = []

        for (inner_train_indexes, inner_valid_indexes), inner_i_fold in iter:
            inner_start = time()

            train_clean_t = [train_clean[i] for i in inner_train_indexes]
            train_clean_v = [train_clean[i] for i in inner_valid_indexes]

            X_train_t, Y_train_t, _ = build_X_Y_wrapper(train_clean_t)
            X_train_v, Y_train_v, _ = build_X_Y_wrapper(train_clean_v)

            model_id = "_".join(["nn", str(i_fold), str(inner_i_fold)])

            Y_model_iter = train_and_predict(
                X_train_t, Y_train_t,
                X_train_v, Y_train_v,
                X_valid, Y_valid,
                inner_i_fold, model_id
            )

            Ys_models.append(Y_model_iter)

            print("├─ Inner fold {}/{}: {}s".format(
                inner_i_fold + 1,
                INNER_VALIDATION,
                round(time() - inner_start, 2)
            ))

        Y_model = ensemble(Ys_models)
    else:
        print("Invalid parameter: INNER_VALIDATION")
        exit()

    ys_model = extract_ys_wrapper(Y_model, valid_lengths, valid_clean)

    for noisy in ys_model:
        y_noisy = file_to_y(noisy)
        y_clean = file_to_y(noisy_to_clean[noisy])
        y_cleaned = ys_model[noisy]

        noisy_filename = filename_from_path(noisy)
        filename, noise_name, snr = noisy_filename.split("|")

        noisy_pesq = pesq_fn(y_clean, y_noisy)
        cleaned_pesq = pesq_fn(y_clean, y_cleaned)

        report.append({
            "noisy_filename": noisy_filename,
            "filename": filename,
            "duration": y_noisy.shape[0] / SAMPLING_RATE,
            "noise_name": noise_name,
            "snr": int(snr),
            "noisy_pesq": noisy_pesq,
            "cleaned_pesq": cleaned_pesq,
            "improved_pesq": cleaned_pesq - noisy_pesq,
            "fold": i_fold + 1
        })

        y_to_file(y_cleaned, efc + noisy_filename + ".wav")

    print("└ {}s".format(round(time() - start, 2)))

columns = ["noisy_filename", "filename", "duration",
           "noise_name", "snr", "noisy_pesq",
           "cleaned_pesq", "improved_pesq", "fold"]

report = pd.DataFrame(report)[columns]
report.to_csv(EXPERIMENT_FOLDER + "report.csv", index=False)

improved_pesq_mean = report["improved_pesq"].mean()
improved_pesq_std = report["improved_pesq"].std()

summary =  "Mean PESQ improvement:   {}\n".format(improved_pesq_mean)
summary += "Stddev PESQ improvement: {}\n".format(improved_pesq_std)
summary += "Mean - 1*Stddev:         {}\n".format(
    improved_pesq_mean - improved_pesq_std
)

print("\n" + summary)

summary_file = open(EXPERIMENT_FOLDER + "summary.txt", "w")
summary_file.write(summary)
summary_file.close()
