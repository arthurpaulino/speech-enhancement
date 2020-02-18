EXPERIMENT_NAME = "exp_test"

########## data generation ##########
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"
SAMPLING_RATE = 8000
NOISE_DB_MULTIPLIERS = [1]

########## data processing ##########
FFT_MS = 32
OVERLAP = 0.5
LOOK_BACK = 2
LOOK_AFTER = 2

########## experiments ##########
RANDOM_SEED = 42
N_EXPERIMENTS = 1
VALIDATION_RATIO = 0.25

########## neural networks ##########
# size multiplier, dropout and activation function
LAYERS = [(3, 0.5, "sigmoid")]
