# Speech Enhancement

This project aims to use NNs to enhance speech quality.

## Main scripts

* `parameters.py` contains the definitions for the data processing and the
experiments. This is the file to tweak.

* `setup_files.py` creates the proper structure of directories for the
experiments. A `generated` folder will be created inside the a `data` folder.
Then, a folder named after `EXPERIMENT_NAME` will be created inside the
`generated` folder.
