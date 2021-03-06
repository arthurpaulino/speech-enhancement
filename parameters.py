EXPERIMENT_NAME = "exp_test"

########## data generation ##########
CLEAN_AUDIO_FOLDER = "data/clean_test/"
NOISES_FOLDER = "data/noise_test/"
NOISY_FOLDER = "data/noisy_test/"
SAMPLING_RATE = 8000
SNRS = [-5] # sound to noise ratio (dB). int values, only

########## reverse learning ##########
REVERSE_LEARNING_FILE = "data/reverse_test.wav"
VOICELESS_INTERVALS = [(63.4, 73.07)]

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
INNER_N_FOLDS = 2
ENSEMBLE_WEIGHTS_POWER = 2

########## neural networks ##########
PATIENCE = 5
MIN_DELTA = 1e-3 # for reverse learning, only
BATCH_SIZE_RATIO = 0.005
VERBOSE = 1
