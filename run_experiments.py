from time import time
import shutil
import json
import os

from sklearn.model_selection import KFold
from numpy.random import seed
import pandas as pd

from parameters import *
from utils import *


validate_pesq()
seed(RANDOM_SEED)

for folder in EXPERIMENT_FOLDER_CLEANED_EXP, EXPERIMENT_FOLDER_MODELS:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

clean_to_noisy = json_load(EXPERIMENT_FOLDER_MAPS + "clean_to_noisy.json")
noisy_to_clean = json_load(EXPERIMENT_FOLDER_MAPS + "noisy_to_clean.json")
audio_to_ampli = json_load(EXPERIMENT_FOLDER_MAPS + "audio_to_ampli.json")
audio_to_phase = json_load(EXPERIMENT_FOLDER_MAPS + "audio_to_phase.json")
audio_to_ampli_eng = json_load(
    EXPERIMENT_FOLDER_MAPS + "audio_to_ampli_eng.json"
)


def build_X_Y_wrapper(clean_list):
    return build_X_Y(clean_list, clean_to_noisy,
                     audio_to_ampli, audio_to_ampli_eng)


def extract_ys_wrapper(Y_model, lengths, clean_list):
    return extract_ys(Y_model, lengths, clean_list,
                      clean_to_noisy, audio_to_phase)


clean_list = list(clean_to_noisy)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
splits = kf.split(clean_list)

report = []

for (train_indexes, valid_indexes), i_fold in zip(splits, range(N_FOLDS)):
    print("\nFold {}/{}".format(i_fold + 1, N_FOLDS))
    start = time()

    train_clean = [clean_list[i] for i in train_indexes]
    valid_clean = [clean_list[i] for i in valid_indexes]

    X_valid, _, valid_lengths = build_X_Y_wrapper(valid_clean)

    inner_kf = KFold(n_splits=INNER_N_FOLDS,
                     shuffle=True,
                     random_state=RANDOM_SEED)

    inner_splits = inner_kf.split(train_clean)

    iter = zip(inner_splits, range(INNER_N_FOLDS))

    Ys_models = []

    for (inner_train_indexes, inner_valid_indexes), inner_i_fold in iter:
        inner_start = time()

        train_clean_t = [train_clean[i] for i in inner_train_indexes]
        train_clean_v = [train_clean[i] for i in inner_valid_indexes]

        X_train_t, Y_train_t, _ = build_X_Y_wrapper(train_clean_t)
        X_train_v, Y_train_v, _ = build_X_Y_wrapper(train_clean_v)

        Y_model_iter = train_and_predict(X_train_t, Y_train_t, X_valid,
                                         X_train_v, Y_train_v)

        Ys_models.append(Y_model_iter)

        print("├─ Inner fold {}/{}: {}m".format(
            inner_i_fold + 1,
            INNER_N_FOLDS,
            round((time() - inner_start) / 60, 2)
        ))

    Y_models = ensemble(Ys_models)

    ys_model = extract_ys_wrapper(Y_models, valid_lengths, valid_clean)

    for noisy in ys_model:
        y_noisy = file_to_y(noisy)
        y_clean = file_to_y(noisy_to_clean[noisy])
        y_cleaned = ys_model[noisy]

        noisy_filename = filename_from_path(noisy)
        filename, noise_name, snr = noisy_filename.split("|")

        noisy_snr = snr_fn(y_clean, y_noisy)
        cleaned_snr = snr_fn(y_clean, y_cleaned)

        noisy_pesq = pesq_fn(y_clean, y_noisy)
        cleaned_pesq = pesq_fn(y_clean, y_cleaned)

        noisy_stoi = stoi_fn(y_clean, y_noisy)
        cleaned_stoi = stoi_fn(y_clean, y_cleaned)

        noisy_estoi = estoi_fn(y_clean, y_noisy)
        cleaned_estoi = estoi_fn(y_clean, y_cleaned)

        report.append({
            "noisy_filename": noisy_filename,
            "filename": filename,
            "duration": y_noisy.shape[0] / SAMPLING_RATE,
            "noise_name": noise_name,
            "snr": int(snr),
            "fold": i_fold + 1,
            "noisy_snr": noisy_snr,
            "cleaned_snr": cleaned_snr,
            "improved_snr": cleaned_snr - noisy_snr,
            "noisy_pesq": noisy_pesq,
            "cleaned_pesq": cleaned_pesq,
            "improved_pesq": cleaned_pesq - noisy_pesq,
            "noisy_stoi": noisy_stoi,
            "cleaned_stoi": cleaned_stoi,
            "improved_stoi": cleaned_stoi - noisy_stoi,
            "noisy_estoi": noisy_estoi,
            "cleaned_estoi": cleaned_estoi,
            "improved_estoi": cleaned_estoi - noisy_estoi
        })

        y_to_file(y_cleaned,
                  EXPERIMENT_FOLDER_CLEANED_EXP + noisy_filename + ".wav")

    print("└ {}m".format(round((time() - start) / 60, 2)))

columns = [
    "noisy_filename", "filename", "duration",
    "noise_name", "snr", "fold",
    "noisy_snr", "cleaned_snr", "improved_snr",
    "noisy_pesq", "cleaned_pesq", "improved_pesq",
    "noisy_stoi", "cleaned_stoi", "improved_stoi",
    "noisy_estoi", "cleaned_estoi", "improved_estoi"
]

report = pd.DataFrame(report)[columns]
report.to_csv(EXPERIMENT_FOLDER + "report.csv", index=False)

improved_snr_mean = report["improved_snr"].mean()
improved_snr_std = report["improved_snr"].std()

improved_pesq_mean = report["improved_pesq"].mean()
improved_pesq_std = report["improved_pesq"].std()

improved_stoi_mean = report["improved_stoi"].mean()
improved_stoi_std = report["improved_stoi"].std()

improved_estoi_mean = report["improved_estoi"].mean()
improved_estoi_std = report["improved_estoi"].std()

summary =  "  SNR: {} ± {} ({})\n".format(
    round(improved_snr_mean, 5),
    round(improved_snr_std, 5),
    round(improved_snr_mean - improved_snr_std, 5)
)

summary += " PESQ: {} ± {} ({})\n".format(
    round(improved_pesq_mean, 5),
    round(improved_pesq_std, 5),
    round(improved_pesq_mean - improved_pesq_std, 5)
)

summary += " STOI: {} ± {} ({})\n".format(
    round(improved_stoi_mean, 5),
    round(improved_stoi_std, 5),
    round(improved_stoi_mean - improved_stoi_std, 5)
)

summary += "ESTOI: {} ± {} ({})\n".format(
    round(improved_estoi_mean, 5),
    round(improved_estoi_std, 5),
    round(improved_estoi_mean - improved_estoi_std, 5)
)

print("\n" + summary)

summary_file = open(EXPERIMENT_FOLDER + "summary.txt", "w")
summary_file.write(summary)
summary_file.close()
