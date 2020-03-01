import os

from parameters import *
from utils import *

if not os.path.exists(EXPERIMENT_FOLDER_REVERSE):
    os.makedirs(EXPERIMENT_FOLDER_REVERSE)

if not isinstance(REVERSE_LEARNING_FILE, str) or\
        not is_valid_audio_file(REVERSE_LEARNING_FILE):
    print("Invalid file:", REVERSE_LEARNING_FILE)
    exit()

y_noisy = file_to_y(REVERSE_LEARNING_FILE)

noisy_filename = filename_from_path(REVERSE_LEARNING_FILE)

y_to_file(
    y_noisy,
    EXPERIMENT_FOLDER_REVERSE + "[noisy]" + noisy_filename + ".wav"
)

y_noisy_len = y_noisy.shape[0]

noisy_abslt, noisy_angle = y_to_abslt_angle(y_noisy)

noisy_abslt_eng = eng_abslt(noisy_abslt)

Xs = []
Ys = []

n_noises = len(VOICELESS_INTERVALS)

for start, end in VOICELESS_INTERVALS:
    y_noise = y_noisy[round(start * SAMPLING_RATE) : round(end * SAMPLING_RATE)]
    y_noise = extend(y_noise, y_noisy_len)

    noise_abslt, _ = y_to_abslt_angle(y_noise)
    Xs.append(noisy_abslt_eng)
    Ys.append(noise_abslt)

X, Y = concatenate(Xs, Ys)

Y_model = train_and_predict(X, Y, noisy_abslt_eng, save_model=False)

y_noisy_pure_noise = abslt_angle_to_y(Y_model, noisy_angle)

y_noisy, y_noisy_pure_noise = cap(y_noisy, y_noisy_pure_noise)

y_cleaned = y_noisy - y_noisy_pure_noise

y_to_file(
    y_noisy_pure_noise,
    EXPERIMENT_FOLDER_REVERSE + "[noise]" + noisy_filename + ".wav"
)

y_to_file(
    y_cleaned,
    EXPERIMENT_FOLDER_REVERSE + "[cleaned]" + noisy_filename + ".wav"
)

print("  SNR:", snr_fn(y_cleaned, y_noisy))
print(" PESQ:", pesq_fn(y_cleaned, y_noisy))
print(" STOI:", stoi_fn(y_cleaned, y_noisy))
print("ESTOI:", estoi_fn(y_cleaned, y_noisy))
