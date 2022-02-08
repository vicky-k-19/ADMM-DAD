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
from matplotlib.ticker import FormatStrFormatter

# import wandb
import librosa
import librosa.display

# For plotting headlessly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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


class UCS(nn.Module):
    def __init__(
        self,
        measurements=400,
        ambient=28 * 28,
        redundancy_multiplier=3,
        sparsity_percentage=0.2,
        admm_iterations=1,
        lamda=0.1,
        rho=1,
        remeasure_y=False,
        sparsity_method="attention",
        measurement_matrix=None,
    ):
        super(UCS, self).__init__()
        print("Model Hyperparameters:")
        print(f"\tmeasurements={measurements}")
        print(f"\tredundancy_multiplier={redundancy_multiplier}")
        print(
            f'\tsparsity_percentage={sparsity_percentage}. Not applicable when sparsity_method="relu_mean"'
        )
        print(f"\tadmm_iterations={admm_iterations}")
        print(f"\tlambda={lamda}")
        print(f"\trho={rho}")
        print(f"\tsparsity_method={sparsity_method}")
        remeasure_y = False  # Force false here, since it's bad formulation if True
        self.lamda = lamda
        self.remeasure_y = remeasure_y
        self.redundancy_multiplier = redundancy_multiplier
        self.sparsity_percentage = sparsity_percentage
        self.admm_iterations = admm_iterations
        self.measurements = measurements
        self.ambient = ambient
        self.activation = ShrinkageActivation()
        if measurement_matrix is None:
            a = torch.randn(measurements, ambient)
        else:
            a = measurement_matrix
        self.register_buffer("a", a)
        # id = torch.eye(self.ambient,self.ambient)
        # idx = torch.randperm(self.measurements)
        # a = id[idx[:self.measurements],:]
        self.rho = rho
        phi = nn.Parameter(self._init_phi())
        self.register_parameter("phi", phi)
        self.sparsity_method = sparsity_method
        if sparsity_method == "attention":
            self.sparsifier = self.sparsify_fx_attention
        elif sparsity_method == "relu_adaptive":
            self.sparsifier = self.sparsify_fx_relu_adaptive
        elif sparsity_method == "relu_mean":
            self.sparsifier = self.sparsify_fx_relu_mean
        elif sparsity_method == "none":
            self.sparsifier = self.dont_sparsify
        else:
            raise ValueError("Unsupported sparsity method")

    def _init_phi(self):
        init = torch.empty(self.ambient * self.redundancy_multiplier, self.ambient)

        init = torch.nn.init.kaiming_normal_(init)

        return init

    def extra_repr(self):
        return "(phi): Parameter({}, {})".format(*self.phi.shape)

    def measure_x(self, x):
        # Create y

        y = torch.einsum("ma,ba->bm", self.a, x)  # (400, 784) * (B,784) -> (B, 400)
        e = torch.randn_like(y)
        y = y + 1e-4 * e

        return y

    def multiplier(self):
        ata = torch.mm(self.a.t(), self.a)  # (784, 400) * (400, 784) -> (784, 784)
        ftf = torch.mm(self.phi.t(), self.phi)
        # m = torch.inverse(ata)
        m = ata + self.rho * ftf
        m_lu, _ = m.lu()
        _, L, U = torch.lu_unpack(m_lu, _)
        Linv = torch.linalg.inv(L)
        Uinv = torch.linalg.inv(U)

        return Linv, Uinv

    def sparsify_fx_relu_adaptive(self, fx, x):
        threshold = torch.quantile(
            fx, 1 - self.sparsity_percentage, dim=-1, keepdim=True
        )  # keeps exactly sparsity_percentage * fx.numel() non zero entries
        fx_sparse = torch.relu(fx - (threshold - 1e-10))  # IT'S MAGIC
        return fx_sparse

    def sparsify_fx_relu_mean(self, fx, x):
        threshold = torch.abs(torch.mean(fx))
        fx_sparse = torch.relu(fx - (threshold - 1e-10))  # IT'S MAGIC
        return fx_sparse

    def dont_sparsify(self, fx, x):
        return fx

    def sparsify_fx_attention(self, fx, x):
        scores = torch.einsum("ba,sa->bs", x, self.phi) / math.sqrt(
            x.size(-1)
        )  # (B, 3*784)
        scores = F.softmax(scores, dim=-1)
        scores = scores.mean(0)  # (3 * 784)
        scores = F.dropout(scores, p=0.1)
        top_scores, indices = torch.topk(
            scores, int(self.sparsity_percentage * scores.size(-1)), dim=-1
        )
        mask = torch.zeros_like(fx)
        mask[:, indices] = 1
        fx_sparse = fx * mask
        return fx_sparse

    def linear(self, x, u):
        fx = torch.einsum("sa,ba->bs", self.phi, x)
        fx = self.sparsifier(fx, x)
        return fx + u  # (B, 3*784)

    def decode(self, y, min_x, max_x, u, z):
        # t1 = 1
        Linv, Uinv = self.multiplier()  # (784,784)
        x0 = torch.einsum("am,bm->ba", self.a.t(), y)
        for _ in range(self.admm_iterations):
            # AF = torch.mm(Uinv,Linv)
            x_L = torch.einsum(
                "aa,ba->ba",
                Linv,
                x0 + torch.einsum("as,bs->ba", self.rho * self.phi.t(), z - u),
            )
            x_hat = torch.einsum("aa,ba->ba", Uinv, x_L)
            fxu = self.linear(x_hat, u)
            z = self.activation(fxu, self.lamda / self.rho)
            u = u + fxu - z

        return torch.clamp(x_hat, min=min_x, max=max_x)
        # return x_hat

    def forward(self, y, x):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)
        u = torch.zeros((x.size(0), self.phi.size(0))).to(y.device)
        z = torch.zeros((x.size(0), self.phi.size(0))).to(y.device)
        x_hat = self.decode(y, min_x, max_x, u, z)

        return x_hat  # (B,784)


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
    input_folder = ARGS.admm_input_folder if algorithm == "admm" else ARGS.ista_input_folder
    ckpt = ARGS.admm_ckpt if algorithm == "admm" else ARGS.ista_ckpt

    evaluation_loader, test_loader, measurement_matrix = get_data_loaders(
        ARGS.batch_size, data_path=input_folder
    )

    if algorithm == "admm":
        model = UCS(
            measurements=ARGS.measurement_factor * ARGS.ambient_dim,
            ambient=AMBIENT_DIM,
            admm_iterations=ARGS.admm_layers,
            lamda=ARGS.lamda,
            rho=ARGS.rho,
            redundancy_multiplier=ARGS.redundancy,
            sparsity_percentage=0.01,
            sparsity_method="none",
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
    stds = np.linspace(0.0,1e-2,50) 
    # stds = [0.0, 1e-5, 1e-2]
    admm_mses = evaluate(stds, algorithm="admm", device=device)
    ista_mses = evaluate(stds, algorithm="ista", device=device)
    print(stds)
    print(admm_mses)
    print(ista_mses)
    fig,ax = plt.subplots()

    #ax.xaxis([0.0,1e-2])

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))

    ax.set_ylabel('MSE',fontsize=15)
    ax.set_xlabel("Noise's standard deviation",fontsize=15)
    plt.yticks(np.linspace(1e-6,3e-4,8))

    ax.plot(stds, admm_mses, marker = '*')
    ax.plot(stds, ista_mses, marker = 'o')
    plt.legend(["10-layer ADMM-DAD","10-layer ISTA-net"], loc = 'center right')
    #ax.plot(stds, ista_mses, linestyle="--")
    plt.show()
    fig.savefig(f"robustness_{ARGS.measurement_factor*800}.png", bbox_inches='tight')