import pickle
import matplotlib.pyplot as plt
from re import sub
import torchaudio
from datasets import (
    MYSPEECHCOMMANDS,
    Device,
    SequenceCollator,
    Timit,
    MeasurementsDataset,
    split_into_windows,
)
import os
import random
import math
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
# import wandb
import librosa
import librosa.display
# For plotting headlessly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import skimage
import skimage.io

TEST_SAMPLE_INDICES_TO_SAVE = [10, 51, 201, 103, 1]


def parse_args():
    parser = argparse.ArgumentParser("Segment and measure speech datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="speechcommands",
        help="which dataset to use [speechcommands|timit]",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Input folder containing segmented and measured data",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Folder to save sample spectrograms"
    )
    parser.add_argument(
        "--measurement-factor",
        type=float,
        default=0.25,
        help="num_measurements=ambient_dim*measurement_factor",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=8000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--ambient-dim",
        type=int,
        default=800,
        help="Ambient dimension. Equal to segment length",
    )

    parser.add_argument(
        "--lamda",
        type=float,
        default=1e-4,
        help="Lamda for the threshold",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=1,
        help="Rho for the threshold",
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=5,
        help="Number of layers/iterations",
    )

    parser.add_argument(
        "--redundancy",
        type=int,
        default=5,
        help="Redundancy factor",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    return parser.parse_args()

DEBUG = False

ARGS = parse_args()

# Model parameters
ADMM_ITERATIONS = ARGS.layers  # Number of ADMM iterations during forward
AMBIENT_DIM = ARGS.ambient_dim
NUM_MEASUREMENTS = round(
    ARGS.measurement_factor*ARGS.ambient_dim
)  # Number of measurements to use for CS
LAMDA = ARGS.lamda  # positive regularization parameter
RHO = ARGS.rho # positive penalty parameter of ADMM
REDUNDANCY_MULTIPLIER = ARGS.redundancy # redundancy ratio of analysis operator

if ARGS.dataset == "speechcommands":
    MAX_LENGTH = ARGS.sample_rate
else:
    MAX_LENGTH = int(60000 * (ARGS.sample_rate / 8000))

CLIP_GRAD_NORM = 10  # Clip gradients to avoid exploding..

#######################################################################################

LEARNING_RATE = ARGS.lr  # Adam Learning rate
BATCH_SIZE = 128  # How many images to process in parallel
NUM_EPOCHS = 50  # Epochs to train


#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################
def get_data_loaders(train_batch_size, val_batch_size, data_path):
    if ARGS.dataset == "speechcommands":
        evaluation = MYSPEECHCOMMANDS(
            root="./data",
            subset="testing",
            sample_rate=ARGS.sample_rate,
            max_length=MAX_LENGTH,
        )
    else:
        evaluation = Timit(
            data_path="./data/timit",
            split="test",
            sample_rate=ARGS.sample_rate,
            max_length=MAX_LENGTH,
        )
    evaluation_loader = DataLoader(
        evaluation,
        num_workers=1,
        batch_size=val_batch_size,
        shuffle=False,
    )
    with open(os.path.join(data_path, "measurement_matrix.p"), "rb") as fd:
        measurement_matrix = pickle.load(fd)
 
    return evaluation_loader, measurement_matrix


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def save_image(grid, fname):
    from PIL import Image

    ndarr = (
        grid.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )

    im = Image.fromarray(ndarr)
    im.save(fname)

    return ndarr

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

 # produce spectrograms and save them

def save_spectrogram(wav_path):
    spec_fn = torchaudio.transforms.Spectrogram(n_fft=1024)
    sig, sr = librosa.load(wav_path)
    #mels = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=2048, hop_length=512)
    mels = spec_fn(torch.tensor(sig))

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    p = librosa.display.specshow(librosa.amplitude_to_db(mels, ref=np.max), ax=ax, sr=sr, y_axis='log', x_axis='time')
    fig.savefig(f'{wav_path}.png')
    ## min-max scale to fit inside 8-bit range
    #img = scale_minmax(mels, 0, 255).astype(np.uint8)
    #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    #img = 255-img # invert. make black==more energy
    print(wav_path)
    print(mels.shape)
    #print(img.shape)
    #skimage.io.imsave(f'{wav_path}.png', img)


def save_examples(model, eval_loader, epoch, algo="admm_classic", device="cpu"):
    folder = ARGS.output_folder
    #spec_fn = torchaudio.transforms.Spectrogram()
    safe_mkdirs(folder)
    idxes = TEST_SAMPLE_INDICES_TO_SAVE
    wavs = [eval_loader.dataset[i][0] for i in idxes]

    segments = [
        torch.stack(
            split_into_windows(
                w, num_windows=int(MAX_LENGTH / AMBIENT_DIM)
            )
        ).to(device)
        for w in wavs
    ]

    segments = [s[s.sum(dim=-1) != 0] for s in segments]
    measurements = [model.measure_x(s) for s in segments]
    reconstructed = [model(m, s) for m, s in zip(measurements, segments)]
    reconstructed = [r.reshape(-1).detach().cpu() for r in reconstructed]
    for idx, (org, rec) in enumerate(zip(wavs, reconstructed)):
        org = org[org != 0.0].unsqueeze(0)
        rec = rec.unsqueeze(0)
        org_path = os.path.join(folder, f"original_{idx}_epoch_{epoch}.wav")
        rec_path = os.path.join(folder, f"reconstructed_{idx}_epoch_{epoch}.wav")
        torchaudio.save(
            org_path,
            org,
            ARGS.sample_rate,
        )
        torchaudio.save(
            rec_path,
            rec,
            ARGS.sample_rate,
        )
        save_spectrogram(org_path)
        save_spectrogram(rec_path)

    return None


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


#######################################################################################
# Model Implementation                                                                #
#######################################################################################


class ShrinkageActivation(nn.Module):
    def __init__(self):
        super(ShrinkageActivation, self).__init__()

    def forward(self, x, lamda):
        return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - lamda)


class DAD(nn.Module):
    def __init__(
        self,
        measurements=200,
        ambient=800,
        redundancy_multiplier=5,
        admm_iterations=5,
        lamda=0.0001,
        rho=1,
        measurement_matrix=None,
    ):
        super(DAD, self).__init__()
        print("Model Hyperparameters:")
        print(f"\tmeasurements={measurements}")
        print(f"\tredundancy_multiplier={redundancy_multiplier}")
        print(f"\tadmm_iterations={admm_iterations}")
        print(f"\tlambda={lamda}")
        print(f"\trho={rho}")
        self.lamda = lamda
        self.rho = rho
        self.redundancy_multiplier = redundancy_multiplier
        self.admm_iterations = admm_iterations
        self.measurements = measurements
        self.ambient = ambient
        self.activation = ShrinkageActivation()
        if measurement_matrix is None:
            a = torch.randn(measurements, ambient)/np.sqrt(self.measurements)
        else:
            a = measurement_matrix
        self.register_buffer("a", a)
        self.rho = rho
        phi = nn.Parameter(self._init_phi())
        self.register_parameter("phi", phi)

    def _init_phi(self):
        # initialization of the analysis operator
        
        init = torch.empty(self.ambient * self.redundancy_multiplier, self.ambient)
        init = torch.nn.init.kaiming_normal_(init)

        return init

    def extra_repr(self):
        return "(phi): Parameter({}, {})".format(*self.phi.shape)

    def measure_x(self, x):
        # Create measurements y_i=Ax_i+noise for each segment x_i of x

        y = torch.einsum("ma,ba->bm", self.a, x)
        n = 1e-4*torch.randn_like(y)
        y = y+n

        return y

    def multiplier(self,rho):
        # m = (A^T*A+Φ^Τ*Φ)^-1
        # Instead of calculating directly the inverse, we take the LU factorization of A^T*A+Φ^Τ*Φ

        ata = torch.mm(self.a.t(), self.a) 
        ftf = torch.mm(self.phi.t(), self.phi)
        m = ata + self.rho * ftf
        m_lu, _ = m.lu()
        _, L, U = torch.lu_unpack(m_lu, _)
        Linv = torch.linalg.inv(L)
        Uinv = torch.linalg.inv(U)

        return Linv, Uinv

    def linear(self, x, u):
        # application of analysis operator Φ
        
        fx = torch.einsum("sa,ba->bs", self.phi, x)
        return fx + u  # (B, 3*784)

    def decode(self, y, min_x, max_x, u, z):
        rho = self.rho
        lamda = self.lamda
        Linv, Uinv = self.multiplier(rho)
        x0 = torch.einsum("am,bm->ba", self.a.t(), y)
            
        for _ in range(self.admm_iterations):
            x_L = torch.einsum("aa,ba->ba",Linv, x0 + torch.einsum("as,bs->ba",rho*self.phi.t(),z-u))
            x_hat = torch.einsum("aa,ba->ba",Uinv,x_L)
            fxu = self.linear(x_hat, u)
            z = self.activation(fxu,lamda/rho)
            u = u + fxu - z

        # truncate the reconstructed x_hat, so that it lies in the same values' interval as the original x
        return torch.clamp(x_hat, min=min_x, max=max_x)

    def forward(self, y, x):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)
        # create the dual variables z, u
        u = torch.zeros((x.size(0), self.phi.size(0))).to(y.device)
        z = torch.zeros((x.size(0), self.phi.size(0))).to(y.device)
        # pass y through the network-decoder to get the output x_hat
        x_hat = self.decode(y, min_x, max_x, u, z)

        return x_hat  # (B,784)


#######################################################################################
# Main                                                                                #
#######################################################################################


if __name__ == "__main__":
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    evaluation_loader, measurement_matrix = get_data_loaders(
        BATCH_SIZE, BATCH_SIZE, data_path=ARGS.input_folder
    )

    model = DAD(
        measurements=NUM_MEASUREMENTS,
        ambient=AMBIENT_DIM,
        admm_iterations=ADMM_ITERATIONS,
        lamda=LAMDA,
        rho=RHO,
        redundancy_multiplier=REDUNDANCY_MULTIPLIER,
        measurement_matrix=measurement_matrix,
    )

    model.load_state_dict(torch.load(ARGS.ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()
    result = save_examples(model, evaluation_loader, "best", algo="admm", device=device)
