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

########## pesq ##########
PESQ_SAMPLING_RATE = 8000 # 8k or 16k
PESQ_MODE = "nb" # "nb" or "wb"

########## experiments ##########
RANDOM_SEED = 43
N_FOLDS = 3
INNER_VALIDATION = 3 # (int > 1) or (0 < float < 1)
ensemble_weights_power = 2

########## neural networks ##########
# size multiplier, activation function and dropout rate
LAYERS = [(2, "relu", 0.2)]

PATIENCE = 5
BATCH_SIZE_RATIO = 0.01
VERBOSE = 0
