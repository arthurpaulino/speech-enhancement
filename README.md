# Speech Enhancement

This project aims to use NNs to enhance speech quality. Since Keras API is being
used, **Python 3.6 is required**.

## Main scripts

* `parameters.py` contains the definitions for the data processing and the
experiments. **This is the file to tweak for optimizations**.

* `process_data.py` first creates the proper structure of directories for the
experiments inside a `data/experiments` folder:

  ```
  data/experiments
  └── EXPERIMENT_NAME
      ├── abslt
      ├── abslt_eng
      ├── angle
      ├── clean
      ├── maps
      └── noisy
  ```

  Where `EXPERIMENT_NAME` is the name given to the experiment on the
  `parameters.py` file. Then, the script populates the internal folders as
  follows:

  1. For each combination of clean audio file, noise file and noise multiplier,
  a noisy audio file will be created inside the `noisy` folder. The respective
  clean audio file will have a copy created inside the `clean` folder;

  2. For each clean and noisy audio, the magnitude and the phase matrices will
  be saved on the `abslt` and `angle` folders, respectively. Noisy audios will
  have their magnitude matrices with engineered columns (look back and look
  after) saved on the `abslt_eng` folder.

  3. Some maps of paths will be saved on the `maps` folder as *json* files,
  namely `audio_to_abslt_eng.json`, `audio_to_angle.json`,
  `noisy_to_clean.json`, `audio_to_abslt.json` and `clean_to_noisy.json`.

* `run_experiments.py` performs the experiments to validate models created with
the data and parameters provided.

* `utils.py` and `neural_networks.py` contain definitions of auxiliary functions
and variables that aim to facilitate the implementation of the other scripts
above. `utils.py` has more general definitions and `neural_networks.py` has
definitions related to NN models.
