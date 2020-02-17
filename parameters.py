EXPERIMENT_NAME = "exp_test"

SAMPLING_RATE = 8000
FFT_MS = 32
OVERLAP = 0.75

LOOK_BACK = 2
LOOK_AFTER = 2

RANDOM_SEED = 42
N_EXPERIMENTS = 1
VALIDATION_RATIO = 0.25

# CLEAN_AUDIO_FOLDER = "data/clean/dev-clean/LibriSpeech/dev-clean/3170/137482/"
# NOISES_FOLDER = "data/noise/"
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"

NOISE_DB_MULTIPLIERS = [1.2, 1, 0.8, 0.6, 0.4]

VALID_AUDIO_EXTENSIONS = ["mp3", "ogg", "wav", "flac", "aac", "wma"]
