EXPERIMENT_NAME = "exp_test"

########## data generation ##########
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"
SAMPLING_RATE = 8000
NOISE_DB_MULTIPLIERS = [1]

########## data processing ##########
FFT_MS = 32
OVERLAP = 0.75
LOOK_BACK = 2
LOOK_AFTER = 2

########## experiments ##########
RANDOM_SEED = 43
N_FOLDS = 3
VALIDATION_RATIO = 0.2

########## neural networks ##########
# size multiplier, activation function and dropout rate
LAYERS = [(3, "sigmoid", 0.2)]

BATCH_SIZE_RATIO = 0.01
VERBOSE = 1
