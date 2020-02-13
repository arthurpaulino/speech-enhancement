EXPERIMENT_NAME = "exp_test"

SAMPLING_RATE = 44100
N_FFT = 1024
OVERLAP = 0.75

PESQ_SAMPLING_RATE = 16000

# CLEAN_AUDIO_FOLDER = "data/clean/dev-clean/LibriSpeech/dev-clean/3170/137482/"
# NOISES_FOLDER = "data/noise/"
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"

NOISE_DB_MULTIPLIERS = [1.2, 1, 0.8, 0.6, 0.4]

VALID_AUDIO_EXTENSIONS = ["mp3", "ogg", "wav", "flac", "aac", "wma"]
