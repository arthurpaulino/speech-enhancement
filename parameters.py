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
N_FOLDS = 2
INNER_VALIDATION = 0.5 # (int > 1) or (0 < float < 1)

########## neural networks ##########
# size multiplier, activation function and dropout rate
LAYERS = [(3, "relu", 0.2)]

BATCH_SIZE_RATIO = 0.01
VERBOSE = 0
