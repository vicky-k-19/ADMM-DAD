import copy
import os
import random
import math
import webbrowser

from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale
from tqdm import tqdm
import argparse

def date_fname():
    uniq_filename = (
        str(datetime.now().date()) + "_" + str(datetime.now().time()).replace(":", ".")
    )

    return uniq_filename

# Pass network's parameter as arguments

def parse_args():
    parser = argparse.ArgumentParser("Network's parameters")
    parser.add_argument(
        "--measurement-factor",
        type=float,
        default=0.25,
        help="num_measurements=ambient_dim*measurement_factor",
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
        "--normalization",
        type=str,
        default="sqrt_m",
        help="Normalization of sensing matrix",
    )
    
    parser.add_argument(
        "--learning-rate",
        type = float,
        default=1e-4,
        help = "Learning rate",
    )

    return parser.parse_args()


DEBUG = False
ARGS = parse_args()
# torch.autograd.set_detect_anomaly(True)

# Model parameters
admm_iterations = ARGS.layers  # Number of ADMM iterations during forward
ambient_dim = 32 * 32  # AMBIENT_DIM -> Vectorized image pixels
num_measurements = round(ARGS.measurement_factor*ambient_dim)  # Number of measurements to use for CS
lamda = ARGS.lamda  # positive regularization parameter
rho = ARGS.rho # positive penalty parameter of ADMM
redundancy_multiplier = ARGS.redundancy # redundancy ratio of analysis operator
normalize = ARGS.normalization # types of normalization for measurement matrix A: 1) None 2) A/sqrt(num_measurements) 3) A*A^T=I

CLIP_GRAD_NORM = 10  # Clip gradients to avoid exploding

#######################################################################################

learning_rate = ARGS.learning_rate  # Adam Learning rate
batch_size = 128  # How many images to process in parallel
num_epochs = 200  # Epochs to train

#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################

def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Normalize((0.5,), (0.5,))])
    train = CIFAR10(download=True, root=".", transform=data_transform, train=True)
    val = CIFAR10(download=False, root=".", transform=data_transform, train=False)
    train_loader = DataLoader(
        train, num_workers=1, batch_size=train_batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val, num_workers=1, batch_size=val_batch_size, shuffle=False
    )

    return train_loader, val_loader


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            log.warning(e)
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

# create folder with saved images
def save_examples(model, val_loader, epoch, algo="admm_cifar", device="cpu"):
    psnr = PSNR()
    folder = f"results_admm_cifar/{algo}_{date_fname()}_epoch.{epoch}"
    safe_mkdirs(folder)
    idxes = random.sample(range(len(val_loader.dataset)), 16)
    original = torch.stack([val_loader.dataset[i][0] for i in idxes]).to(device)
    reconstructed = model(original).view(original.size(0), 1, 32, 32)
    original = original.detach().cpu()
    reconstructed = reconstructed.detach().cpu()
    # calculate the PSNR for the saved example
    mypsnr = psnr(original,reconstructed)
    print(f"PSNR = {mypsnr}")
    img_orig = torchvision.utils.make_grid(original, nrow=4)
    img_recon = torchvision.utils.make_grid(reconstructed, nrow=4)
    original_image = save_image(img_orig, f"{folder}/original_epoch_{epoch}.jpg")
    reconstructed_image = save_image(img_recon, f"{folder}/reconstructed_epoch_{epoch}.jpg")
    html = """
    <!DOCTYPE html>
    <html>
    <body>

    <h2>Original</h2>
    <img src="original_epoch_{epoch}.jpg" width="500" height="500">


    <h2>Reconstructed</h2>
    <img src="reconstructed_epoch_{epoch}.jpg" width="500" height="500">

    </body>
    </html>    
    """.format(
        epoch=epoch
    )
    html_file = f"{folder}/epoch_{epoch}.html"
    with open(html_file, "w") as fd:
        fd.write(html)
    # webbrowser.open(html_file)

    return {
        "original": original_image,
        "reconstructed": reconstructed_image,
        "original_caption": f"Original",
        "reconstructed_caption": f"Reconstructed"
    }

class PSNR:
    # Peak Signal to Noise Ratio for img1 and img2 with maximum pixel value = 1

    def __init__(self):
        self.name = "PSNR"

    def __call__(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

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
        ambient=32 * 32,
        redundancy_multiplier=5,
        admm_iterations=10,
        lamda=0.0001,
        rho=1,
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
        a = torch.randn(measurements, ambient) 
        if normalize == None:
            a = a
        elif normalize == "sqrt_m":
            a = a/np.sqrt(self.measurements)
        elif normalize == "orth":
            a = torch.nn.init.orthogonal_(a.t()).t().contiguous()
        self.register_buffer("a", a)
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
        # Create measurements y=Ax+noise

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
        return fx + u  # (B, 3*784)

    def decode(self, y, min_x, max_x,u,z):
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

    def forward(self, x):
        x = x.view(x.size(0), -1)
        min_x = torch.min(x)
        max_x = torch.max(x)
        # measure x
        y = self.measure_x(x)
        # apply analysis operator Φ
        fx = torch.einsum("sa,ba->bs", self.phi, x)
        # create the dual variables z, u
        u = torch.zeros_like(fx)
        z = torch.zeros_like(fx)
        # pass y through the network-decoder to get the output x_hat
        x_hat = self.decode(y,min_x,max_x,u,z)

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
    x_original, _ = batch
    x_original = x_original.to(device)

    def compute_loss():
        x_pred = model(x_original)
        mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

# we separately keep loss and mse, in case a regularizer is added; in the that case, we would have loss = mse + reg

        loss = mse
        return loss

    x_pred = model(x_original)
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

    for batch in train_iter:
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
                "loss": "{:.3}".format(avg_train_loss / n_proc),
            }
        )
        # break
    avg_train_loss = avg_train_loss / len(train_loader)
    avg_train_mse = avg_train_mse / len(train_loader)

    return avg_train_loss, avg_train_mse


def val_step(model, criterion, batch, device="cpu"):
    x_original, _ = batch
    x_original = x_original.to(device)
    x_pred = model(x_original)
    mse = criterion(x_pred, x_original.view(x_original.size(0), -1))

    return mse


def val_epoch(model, criterion, val_loader, device="cpu"):
    avg_val_mse = 0
    n_proc = 0
    val_iter = tqdm(val_loader, desc="test", leave=False)

    model.eval()

    for batch in val_iter:
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
    epochs,
    device="cpu",
):

    for e in range(epochs):
        avg_train_loss, avg_train_mse = train_epoch(
            model, optimizer, criterion, train_loader, device=device
        )
        avg_val_mse = val_epoch(model, criterion, val_loader, device=device)
        result = save_examples(model, val_loader, e, algo="admm", device=device)
        gen_mse = np.abs(avg_train_mse-avg_val_mse)
        
        # print in each epoch the average train+test MSEs and the generalization error
        print({"Epoch": e, "Train MSE": avg_train_mse, "Test MSE": avg_val_mse,})
        print("--------------------------------------")
        print("Average Train MSE = {:.4f}".format(avg_train_mse))
        print("--------------------------------------")
        print("Average Test MSE = {:.4f}".format(avg_val_mse))
        print("--------------------------------------")
        print("Average generalization error = {:.6f}".format(gen_mse))
        print("--------------------------------------")
        print("epoch: ", e)
        print("--------------------------------------")


#######################################################################################
# Main                                                                                #
#######################################################################################


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_data_loaders(batch_size, batch_size)

    model = DAD(
        measurements=num_measurements,
        ambient=ambient_dim,
        admm_iterations=admm_iterations,
        lamda=lamda,
        rho=rho,
        redundancy_multiplier=redundancy_multiplier,
    ).to(device)

    print(model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    epochs = num_epochs

    train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        epochs,
        device=device,
    )
