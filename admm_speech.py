import pickle
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
import matplotlib
import matplotlib.pyplot as plt

# Pass network's parameter as arguments

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
        default=1e-5,
        help="Learning rate",
    )

    return parser.parse_args()


def date_fname():
    uniq_filename = (
        str(datetime.now().date()) + "_" + str(datetime.now().time()).replace(":", ".")
    )

    return uniq_filename


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
    with open(os.path.join(data_path, "measurement_matrix.p"), "rb") as fd:
        measurement_matrix = pickle.load(fd)
    train = MeasurementsDataset(data_path, split="train")
    val = MeasurementsDataset(data_path, split="test")
    train_loader = DataLoader(
        train, num_workers=2, batch_size=train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val, num_workers=2, batch_size=val_batch_size, shuffle=False
    )
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
    return train_loader, val_loader, evaluation_loader, measurement_matrix


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


def save_examples(model, eval_loader, epoch, algo="admm_speech", device="cpu"):
    folder = f"results_speech_admm/{algo}_{date_fname()}_epoch.{epoch}"
    #spec_fn = torchaudio.transforms.Spectrogram()
    safe_mkdirs(folder)
    idxes = random.sample(range(len(eval_loader.dataset)), 4)
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
        torchaudio.save(
            os.path.join(folder, f"original_{idx}_epoch_{epoch}.wav"),
            org.unsqueeze(0),
            ARGS.sample_rate,
        )
        torchaudio.save(
            os.path.join(folder, f"reconstructed_{idx}_epoch_{epoch}.wav"),
            rec.unsqueeze(0),
            ARGS.sample_rate,
        )

    return None


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


#######################################################################################
# Model Implementation                                                                #
#######################################################################################


class ShrinkageActivation(nn.Module):
    def __init__(self):
        super(ShrinkageActivation, self).__init__()

    # implements the softh-thresholding function employed in ADMM
    def forward(self, x, lamda):
        return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - lamda)

# Definition of the decoder
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


#######################################################################################
# Training Functions                                                                  #
#######################################################################################


def train_step(
    model,
    optimizer,
    criterion,
    batch,
    device="cpu",
):
    optimizer.zero_grad()
    x_measurement, x_original = batch
    x_original = x_original.to(device)
    x_measurement = x_measurement.to(device)

    def compute_loss():
        x_pred = model(x_measurement, x_original)
        mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

# we separately keep loss and mse, in case a regularizer is added; in the that case, we would have loss = mse + reg
        
        loss = mse
        return loss

    x_pred = model(x_measurement, x_original)
    mse = criterion(x_pred, x_original.view(x_original.size(0), -1))
    loss = mse
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
    # optimizer.step(compute_loss)
    optimizer.step()
    return loss, mse


def train_epoch(
    model,
    optimizer,
    criterion,
    train_loader,
    device="cpu",
):
    avg_train_loss = 0
    avg_train_mse = 0
    n_proc = 0

    train_iter = tqdm(train_loader, desc="training", leave=False)

    model.train()

    for idx, batch in enumerate(train_iter):
        n_proc += 1
        loss, mse = train_step(
            model,
            optimizer,
            criterion,
            batch,
            device=device,
        )
        avg_train_loss += loss.item()
        avg_train_mse += mse.item()
        train_iter.set_postfix(
            {
                "loss": "{:.4}".format(avg_train_loss / n_proc),
            }
        )
        # break
    avg_train_loss = avg_train_loss / len(train_loader)
    avg_train_mse = avg_train_mse / len(train_loader)

    return avg_train_loss, avg_train_mse


def val_step(model, criterion, batch, device="cpu"):
    x_measurement, x_original = batch
    x_original = x_original.to(device)
    x_measurement = x_measurement.to(device)
    x_pred = model(x_measurement, x_original)
    mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

    return mse


def val_epoch(model, criterion, val_loader, device="cpu"):
    avg_val_mse = 0
    n_proc = 0
    val_iter = tqdm(val_loader, desc="test", leave=False)

    model.eval()

    for idx, batch in enumerate(val_iter):
        n_proc += 1
        mse = val_step(model, criterion, batch, device=device)
        avg_val_mse += mse.item()
        val_iter.set_postfix(
            {
                "test_mse": "{:.3}".format(avg_val_mse / n_proc),
            }
        )
        # break
    avg_val_mse = avg_val_mse / len(val_loader)

    return avg_val_mse


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    evaluation_loader,
    epochs,
    checkpoint_name,
    device="cpu",
):
    
    best_mse = 1e10
    patience = 3

    for e in range(epochs):
        avg_train_loss, avg_train_mse = train_epoch(
            model, optimizer, criterion, train_loader, device=device
        )
        avg_val_mse = val_epoch(model, criterion, val_loader, device=device)

        result = save_examples(model, evaluation_loader, e, algo="admm", device=device)
        gen_mse = np.abs(avg_train_mse-avg_val_mse)

        print({"Epoch": e, "Train MSE": avg_train_mse, "Test MSE": avg_val_mse,})
        print("--------------------------------------")
        print("Average Train MSE = {:.20f}".format(avg_train_mse))
        print("--------------------------------------")
        print("Average Test MSE = {:.20f}".format(avg_val_mse))
        print("--------------------------------------")
        print("Average generalization error = {:.20f}".format(gen_mse))
        print("--------------------------------------")
        print("epoch: ", e)
        print("--------------------------------------")

        if avg_val_mse < best_mse:
            print("Current best Test MSE = {:.20f}".format(avg_val_mse))
            torch.save(model.state_dict(), checkpoint_name)
            patience = 3
        else:
            patience -= 1

            if patience == 0:
                print(f"Stopping at epoch {e}")
                break


#######################################################################################
# Main                                                                                #
#######################################################################################


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, evaluation_loader, measurement_matrix = get_data_loaders(
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
    ).to(device)


    print(model)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    epochs = NUM_EPOCHS

    checkpoint_name = f"admm-{ARGS.dataset}-l{ARGS.layers}-mfactor{ARGS.measurement_factor}-lr{ARGS.lr}-rho{ARGS.rho}.pt"

    train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        evaluation_loader,
        epochs,
        checkpoint_name,
        device=device,
    )
