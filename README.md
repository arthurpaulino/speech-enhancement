# Speech Enhancement

This project aims to use neural networks (NNs) to enhance speech quality of
noisy recordings. Since Keras API is being used, **Python 3.6 is required**.

## The plan of action

The key idea is to train NNs that map noisy spectrograms into clean ones. Thus,
we need a set of clean files and a set of noises, then we generate artificially
noisy files and serve their spectrograms as input to the NNs, for which the
output is the spectrogram of the respective clean file.

On the paper *Causal Speech Enhancement Combining Data-driven Learning and
Suppression Rule Estimation* by Seyedmahdad Mirsamadi and Ivan Tashev, some NN
architectures were proposed to solve this problem on an online (causal) context.
They also tested such architectures on an offline (symmetric) context. The main
difference between online and offline here is that, whereas a NN for the offline
context can have access to some frames ahead of the target frame, a NN for the
online context cannot. Examples of offline and online contexts are recordings
and real-time conversations respectively.

The best results for the offline context were (luckily) achieved by the simplest
NN architecture, which consists of a single dense hidden layer. We intend to
improve their solution for the offline context with new heuristics. A big piece
of this puzzle is that spectrograms can be treated as images. So, convolutional
layers mixed with max pooling layers can be used to extract higher level
features and help the NNs reconstruct clean spectrograms.

Although NNs need a differentiable loss function to minimize, the validation of
the optimization attempts will be done with real audio metrics:

* SNR (Sound-to-Noise Ratio)
* PESQ (Perceptual Evaluation of Speech Quality)
* STOI (Short-Time Objective Intelligibility)
* ESTOI (Extended Short-Time Objective Intelligibility)

This is a common practice to ensure that the predictions of trained models
actually translate into audios with better quality.

## Data pipeline

We begin with a digital representation of audio: a sequence of air pressure
samplings captured at a certain sampling rate, usually measured in kilohertz
(kHz). Then, with a mathematical process called "Short-time Fourier transform",
we can extract the fundamental waves from small sequences of samplings, that is,
a set of sine-like waves that add up to the original one.

A fundamental wave is so simple that it only needs three real numbers to
represent it: frequency, amplitude and phase. We say that we can extract a *set*
of fundamental waves because we get, for each frequency, its amplitude and its
phase. Of course there's no such thing as "each frequency" for the domain of
frequencies is the real numbers. But let's abstract that and move on with the
idea that this process is done for a sufficiently large set of well defined
frequencies.

Numerically speaking, however, the pair amplitude+phase is represented as a
complex number whose absolute value is the amplitude and whose angle is the
phase.

```
c = a + b * i

      |
      b...c (The absolute value is the distance from the origin to this point)
      |  /.
      | / .
      |/ (Angle)
──────|───a────
      |
```

We have almost everything we need to achieve a rich and structured numerical
representation of audio. The last piece of the puzzle is how to extract the data
from a *longer* audio given that we have a tool to deal with *short* chunks of
air pressure samplings. The answer is to partition longer audios in overlapping
and shorter pieces (frames). Overlapping avoids eventual "hiccups" that may
happen due to unlucky numerical leaps between consecutive frames.

## The proposed architecture

On a typical image classification problem, we want to use convolution to extract
information from the images and use solely these higher level features to build
a classifier. Unfortunately, this strategy is prone to underfitting on a
reconstruction/enhancement problem. However, it's still possible to make use of
convolutional layers in order to achieve better results.

The proposed architecture is as follows:

```
           ┌─────┐
┌─┐        |dense|                ┌─┐
|i| ╔═════>|layer|═════╗ ┌─────┐  |o|
|n|═╝      └─────┘     ╚>|dense|  |u|
|p|      ╔══════════════>|     |═>|t|
|u|═╗ ┌──╨──┐  ┌─────┐ ╔>|layer|  |p|
|t| ╚>|conv |═>|conv |═╝ └─────┘  |u|
└─┘   |layer|  |layer|            |t|
      └─────┘  └─────┘            └─┘
```

The second dense layer can use information from the first one and from the
convolutional layers before outputting the predictions.

## Main scripts

* `parameters.py` contains the definitions for the data processing and the
experiments.

* `process_data.py` first creates the proper structure of directories for the
experiments inside a `data/experiments` folder:

  ```
  data/experiments
  └── EXPERIMENT_NAME
      ├── ampli
      ├── ampli_eng
      ├── phase
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
  be saved on the `ampli` and `phase` folders, respectively. Noisy audios will
  have their magnitude matrices with engineered columns (peeking previous and
  incoming rows) saved on the `ampli_eng` folder.

  3. Some maps of paths will be saved on the `maps` folder as *json* files,
  namely `audio_to_ampli_eng.json`, `audio_to_phase.json`,
  `noisy_to_clean.json`, `audio_to_ampli.json` and `clean_to_noisy.json`.

* `run_experiments.py` performs the experiments to validate models created with
the data and parameters provided. An experiment consists of a cross-validation
on the clean files. Every fold of the cross-validation uses part of the training
data as an early stop validator and runs an inner k-fold split. The spectrograms
generated by each of the models will be ensembled into a resulting spectrogram.

  This script creates two folders inside the one related to the experiment:
  `models`, the target folder for the trained models, and `cleaned_exp`, where
  the audio files that were cleaned by the models during the experiment are
  placed.

  It also creates a full report of the experiment on a file called `report.csv`
  and a brief summary of the experiment on a file `summary.txt` containing the
  mean improvements of each audio metric. Both files are located at the
  experiment folder.

* `clean_noisy_files.py` applies the models on the files placed at
`NOISY_FOLDER`. Copies of the noisy files are created at
`data/experiments/EXPERIMENT_NAME/noisy` and the denoised files are saved at
`data/experiments/EXPERIMENT_NAME/clean`. The cleaning process uses an ensemble
of the models created by the `run_experiments.py` script.

* `utils.py` contains definitions of auxiliary functions and variables that aim
to facilitate the implementation of the other scripts above.

## Ensembling spectrograms

Given a set of predicted spectrograms, instead of simply computing the average
magnitude of each spectogram cell, we can perform a smarter weighted average
that penalizes values that are further away from the consensus.
