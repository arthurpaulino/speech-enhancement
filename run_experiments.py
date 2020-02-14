from random import shuffle, seed
import json

from parameters import *
from utils import *


seed(RANDOM_SEED)

clean_to_noisy = json_load(
    "data/generated/" + EXPERIMENT_NAME + "/clean_to_noisy.json"
)

noisy_to_clean = json_load(
    "data/generated/" + EXPERIMENT_NAME + "/noisy_to_clean.json"
)

clean_list = list(clean_to_noisy)

split = round(VALIDATION_RATIO * len(clean_list))

for i_experiment in range(N_EXPERIMENTS):
    shuffle(clean_list)
    valid_clean = clean_list[:split]
    train_clean = clean_list[split:]
