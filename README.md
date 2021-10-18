# ADMM-DAD net

Code for the experiments of the paper "ADMM-DAD net: a deep unfolding network for analysis compressed sensing", V. Kouni, G. Paraskevopoulos, H. Rauhut, G. C. Alexandropoulos, arXiv preprint, arXiv: 2110.06986. 

The repository contains three main scripts for ADMM-DAD: `admm_mnist.py`, `admm_cifar.py`, `admm_speech.py`.

The first two scripts train the model on the MNIST and CIFAR10 datasets respectively and create a folder named `results_admm_dataset` (`dataset=MNIST or CIFAR10`) containing examples of an original and reconstructed image from the corresponding dataset.

The `admm_speech.py` a) trains the model on the SpeechCommands and TIMIT datasets, depending on the choice of the user b) creates a folder named `results_speech_admm` containing examples of original and reconstructed raw speech samples from the corresponding dataset c) produces a checkpoint of the trained model. With this checkpoint, the user can run `admm_extract_test_spectrograms.py` to extract spectrograms of an example test raw audio file of TIMIT. 

Some additional scripts in the repository are `measure_speech.py`, `measure_clean.py` and `evaluate_robustness.py`.

`measure_speech.py` implements the preprocessing on the raw speech samples of the whole SpeechCommands and TIMIT datasets, depending on the choice of the user.

`measure_clean.py` creates noiseless measurements of the raw speech samples of the whole SpeechCommands and TIMIT datasets, depending on the choice of the user.

`evaluate_robustness.py` takes checkpoints of ADMM-DAD net and ISTA-net and plots the desired robustness graph.

# How to run MNIST/CIFAR10

Run 

```
python admm_mnist.py --measurement-factor s --lamda 1e-4 --rho 1 --layers 5 --redundancy 5 --normalization NORMALIZE --learning-rate 1e-4

```

to train the model with MNIST (similarly with CIFAR10). `s` is a CS ratio in {0.25, 0.40, 0.50} and `NORMALIZE` stands for the type of desired normalization to be applied on the measurement matrix A (None for A, sqrt_m for A/sqrt(num_measurements), orth for AA^T=I).


# How to run SpeechCommands

### Download dataset
Set `DOWNLOAD_DATA=True` in `measure_speech.py` or create a folder `data` in your working directory, then extract `speech_commands_v0.02.tar.gz` into `data/SpeechCommands/speech_commands_v0.02`.

### Segment data and obtain measurements

Run 

```
python measure_speech.py --dataset speechcommands --measurement-factor s --ambient-dim 800 --sample-rate 8000 --normalization NORMALIZE,

```
with s and NORMALIZE defined as previously.

This will create a folder in `data/speechcommands_s_800_8000_NORMALIZE`.

The contents of this folder are:

```
├── measurement_matrix.p  # Pickle containing Measurement matrix A
├── test.lmdb  # Test data lmdb file
├── test.lmdb-lock
├── train.lmdb  # Train data lmdb file
├── train.lmdb-lock
├── val.lmdb  # Validation data lmdb file
└── val.lmdb-lock
```

### Train model

Run 

```
python admm_speech.py --input-folder data/speechcommands_s_800_8000_NORMALIZE --ambient-dim 800 --measurement-factor s --lamda 1e-4 --rho 1 --layers 5 --redundancy 5 --lr 1e-5
```

to train the model.


# How to run Timit

### Download data

Get the tarball from 
```
https://drive.google.com/file/d/1Co7I_sWqQFVl0t39fXnBnAmZhV4E1tcd/view?usp=sharing
```

Then move timit.tgz into the `data` folder and run

```
tar xvf timit.tgz
```

### Segment, measure and train


Run

```
python measure_speech.py --dataset timit --ambient-dim ...

```

to segment and measure data. It will create a folder `data/timit_{NUM_MEASUREMENTS}_{AMBIENT_DIM}_{SAMPLE_RATE}`.


```
python admm_speech.py --input_folder data/timit_200_800_8000 ...

```

to train the model.


# Extract Spectrograms

Train a model with `admm_speech.py` to save a checkpoint and then run 

```
python admm_extract_test_spectrograms.py --dataset timit --input-folder data/timit_400_800_8000_sqrt_m/ --measurement-factor 0.5 --sample-rate 8000 --ambient-dim 800 --ckpt chechpoint_name.pt  --output-folder my_output_folder

```

# Create robustness plot

For 40% CS ratio (respectively for 25%), run

```
python evaluate_robustness.py --dataset timit --ista-input-folder data/timit_320_800_8000_ISTA/ --admm-input-folder data/timit_320_800_8000_sqrt_m/ --ista-ckpt ${HOME}/checkpoints/timit_dista/deepISTA-timit-L10-0.001-320.pt --admm-ckpt ${HOME}/checkpoints/timit_admm/admm-timit-l10-mfactor0.4-lr1e-05-rho1.0.pt --output-folder timit_robustness_specs_320_10-10L --measurement-factor 0.4 --sample-rate 8000 --ambient-dim 800 --admm-layers 10 --ista-layers 10

```
where `deepISTA` is the script implementing ISTA-net.

The scripts implementing the baseline ISTA-net were provided to us by original authors of "Compressive Sensing and Neural Networks from a Statistical Learning Perspective", A. Behboodi, H. Rauhut, E. Schnoor, arXiv preprint, arXiv:2010.15658 (2020). For reproducibility purposes, the interested reader may contact them. 
