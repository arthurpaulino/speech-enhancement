EXPERIMENT_NAME = "exp_test"

########## data generation ##########
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"
NOISY_FOLDER = "data/noisy_test/"
SAMPLING_RATE = 8000
SNRS = [-5] # sound to noise ratio (dB). int values, only

########## data processing ##########
FFT_MS = 32    # length of each frame in milliseconds
OVERLAP = 0.75 # the % of overlap between frames
PEEK = 2       # peek rows above and below then append them to the current row

########## pesq ##########
PESQ_SAMPLING_RATE = 8000 # 8k or 16k
PESQ_MODE = "nb" # "nb" or "wb"

########## experiments ##########
RANDOM_SEED = 43
N_FOLDS = 2
INNER_N_FOLDS = 2
ENSEMBLE_WEIGHTS_POWER = 2

########## neural networks ##########
PATIENCE = 5
BATCH_SIZE_RATIO = 0.005
VERBOSE = 1
