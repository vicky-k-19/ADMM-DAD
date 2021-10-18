import argparse
import math
import os
import pickle
import random
from datetime import datetime
from re import sub

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
# For plotting headlessly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

# import wandb
import librosa
import librosa.display
import torchaudio
from datasets import (MYSPEECHCOMMANDS, Device, MeasurementsDataset,
                      SequenceCollator, Timit, split_into_windows)
from utils import Unfolded_ST

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
        "--admm-input-folder",
        type=str,
        help="Input folder containing segmented and measured data",
    )
    parser.add_argument(
        "--ista-input-folder",
        type=str,
        help="Input folder containing segmented and measured data",
    )
    parser.add_argument(
        "--admm-ckpt",
        type=str,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--ista-ckpt",
        type=str,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--output-folder", type=str, help="Folder to save sample spectrograms"
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
        "--admm-layers",
        type=int,
        default=5,
        help="Number of layers/iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--ista-layers",
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

    return parser.parse_args()


DEBUG = False

ARGS = parse_args()

# Model parameters
ISTA_THRESHOLD = 1e-4
AMBIENT_DIM = ARGS.ambient_dim

if ARGS.dataset == "speechcommands":
    MAX_LENGTH = ARGS.sample_rate
else:
    MAX_LENGTH = int(60000 * (ARGS.sample_rate / 8000))

#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################

def get_data_loaders(batch_size, data_path):
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
        batch_size=batch_size,
        shuffle=False,
    )
    with open(os.path.join(data_path, "measurement_matrix.p"), "rb") as fd:
        measurement_matrix = pickle.load(fd)

    test = MeasurementsDataset(data_path, split="test.clean")
    test_loader = DataLoader(
        test, num_workers=8, batch_size=batch_size, shuffle=False
    )
 
    return evaluation_loader, test_loader, measurement_matrix


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


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
        redundancy_multiplier=3,
        admm_iterations=10,
        lamda=0.1,
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
        ftf = torch.mm(self.phi.t(),self.phi)
        m = ata + rho * ftf
        m_lu, _ = m.lu()
        _, L, U = torch.lu_unpack(m_lu, _)
        Linv = torch.linalg.inv(L)
        Uinv = torch.linalg.inv(U)

        return Linv, Uinv

    def linear(self, x, u):
        # application of analysis operator Φ

        fx = torch.einsum("sa,ba->bs", self.phi, x)
        return fx + u 

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
        return torch.clamp(x_hat,min=min_x,max=max_x)      

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

        return x_hat 

def save_spectrogram(wav_path):
    spec_fn = torchaudio.transforms.Spectrogram(n_fft=1024)
    sig, sr = librosa.load(wav_path)
    # mels = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=2048, hop_length=512)
    mels = spec_fn(torch.tensor(sig))

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    p = librosa.display.specshow(
        librosa.amplitude_to_db(mels, ref=np.max),
        ax=ax,
        sr=sr,
        y_axis="log",
        x_axis="time",
    )
    fig.savefig(f"{wav_path}.png")


def insert_noise(y, std=0.0):
    n = torch.randn_like(y) * std

    return y + n


def measure_x(model, x, algorithm="admm", std=0.0, device="cpu"):
    if algorithm == "ista":
        y = (model.measurement @ x.t()).T.to(device)
    else:
        y = model.measure_x(x)

    y = insert_noise(y, std=std)

    return y


def reconstruct(model, y, x, algorithm="admm", device="cpu"):
    if algorithm == "ista":
        x_rec = model(y, x, ISTA_THRESHOLD)
    else:
        x_rec = model(y, x)

    return x_rec


def save_examples(model, eval_loader, algorithm="admm", std=0.0, device="cpu"):
    folder = ARGS.output_folder
    safe_mkdirs(folder)
    idxes = TEST_SAMPLE_INDICES_TO_SAVE
    wavs = [eval_loader.dataset[i][0] for i in idxes]

    segments = [
        torch.stack(
            split_into_windows(w, num_windows=int(MAX_LENGTH / AMBIENT_DIM))
        ).to(device)

        for w in wavs
    ]

    segments = [s[s.sum(dim=-1) != 0] for s in segments]
    measurements = [measure_x(model, s, algorithm=algorithm, std=std, device=device) for s in segments]
    reconstructed = [
        reconstruct(model, m, s, algorithm=algorithm, device=device)

        for m, s in zip(measurements, segments)
    ]
    reconstructed = [r.reshape(-1).detach().cpu() for r in reconstructed]

    for idx, (org, rec) in enumerate(zip(wavs, reconstructed)):
        org = org[org != 0.0].unsqueeze(0).detach().cpu()
        rec = rec.unsqueeze(0).detach().cpu()
        org_path = os.path.join(folder, f"original_{idx}.wav")
        rec_path = os.path.join(folder, f"reconstructed_{idx}_{algorithm}_std_{std}.wav")
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


def run_robustness(model, test_loader, criterion, evaluation_loader, algorithm="admm", std=0.0, device="cpu"):
    model.eval()
    n_proc = 0
    avg_val_mse = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc=f"robustness test std={std}")):
            n_proc += 1
            y, x  = batch
            y = insert_noise(y, std=std)
            y = y.to(device).squeeze()
            x = x.to(device)
            x_pred = reconstruct(model, y, x, algorithm=algorithm, device=device)
            mse = criterion(x_pred, x.view(x.size(0), -1))
            avg_val_mse += mse.item()

    avg_val_mse = avg_val_mse / n_proc
    print(f"Robustness test std={std} | MSE={avg_val_mse:.20}")

    save_examples(model, evaluation_loader, std=std, algorithm=algorithm, device=device)

    return avg_val_mse

#######################################################################################
# Main                                                                                #
#######################################################################################


def evaluate(stds, algorithm="admm", device="cpu"):
    """Run robustness tests. Evaluate for different levels of measurement noise."""
    input_folder = ARGS.admm_input_folder if algorithm == "admm" else ARGS.ista_input_folder
    ckpt = ARGS.admm_ckpt if algorithm == "admm" else ARGS.ista_ckpt

    evaluation_loader, test_loader, measurement_matrix = get_data_loaders(
        ARGS.batch_size, data_path=input_folder
    )

    if algorithm == "admm":
        model = DAD(
            measurements=ARGS.measurement_factor * ARGS.ambient_dim,
            ambient=AMBIENT_DIM,
            admm_iterations=ARGS.admm_layers,
            lamda=ARGS.lamda,
            rho=ARGS.rho,
            redundancy_multiplier=ARGS.redundancy,
            measurement_matrix=measurement_matrix,
        )
    else:
        model = Unfolded_ST(ARGS.ista_layers, measurement_matrix, device=device)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()

    mses = [
        run_robustness(
            model,
            test_loader,
            criterion,
            evaluation_loader,
            algorithm=algorithm,
            std=std,
            device=device
        ) for std in stds
    ]

    return mses


if __name__ == "__main__":
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    stds = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    admm_mses = evaluate(stds, algorithm="admm", device=device)
    ista_mses = evaluate(stds, algorithm="ista", device=device)
    print(stds)
    print(admm_mses)
    print(ista_mses)
    fig,ax = plt.subplots()

    ax.set_ylabel('MSE')
    ax.set_xlabel("Noise standard deviation")
    
    ax.plot(stds, admm_mses)
    ax.plot(stds, ista_mses, linestyle="--")
    fig.savefig("robustness.png")
