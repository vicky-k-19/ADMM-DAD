import argparse
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import lmdb
from datasets import MYSPEECHCOMMANDS, Timit, pad_tensor, split_into_windows


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def parse_args():
    parser = argparse.ArgumentParser("Segment and measure speech datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="speechcommands",
        help="which dataset to use [speechcommands|timit]",
    )
    parser.add_argument(
        "--measurement-factor",
        type=float,
        default=0.25,
        help="num_measurements=ambient_dim*measurement_factor",
    )
    parser.add_argument(
        "--ambient-dim",
        type=int,
        default=800,
        help="Ambient dimension. Equal to segment length",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=8000, help="Audio sample rate"
    )
    parser.add_argument(
        "--noise-padding",
        action="store_true",
        help="Use gaussian noise for padding instead of zero padding",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download speech commands dataset",
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        default=None,
        help="Output folder to store the dataset",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        help="Normalization on measurement matrix",
    )

    return parser.parse_args()


ARGS = parse_args()


AMBIENT_DIM = ARGS.ambient_dim
MEASUREMENT_FACTOR = ARGS.measurement_factor
NUM_MEASUREMENTS = round(MEASUREMENT_FACTOR*AMBIENT_DIM)
SAMPLE_RATE = ARGS.sample_rate

if ARGS.dataset == "speechcommands":
    MAX_LENGTH = SAMPLE_RATE  # Speech commands contains ~1sec segments
elif ARGS.dataset == "timit":
    MAX_LENGTH = int(
        60000 * (SAMPLE_RATE / 8000)
    )  # For sr == 8000 -> max length = 60000. SR=16000 -> MAX_LENGTH=120000
PADDING = "constant"  # or "gaussian" or "uniform"
DOWNLOAD_DATA = ARGS.download
NORMALIZE = ARGS.normalization

#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################

def get_data_loaders(dataset="speechcommands"):
    if dataset == "speechcommands":
        train = MYSPEECHCOMMANDS(
            root="./data",
            subset="training",
            sample_rate=SAMPLE_RATE,
            download=DOWNLOAD_DATA,
        )
        val = MYSPEECHCOMMANDS(
            root="./data", subset="validation", sample_rate=SAMPLE_RATE, download=False
        )
        test = MYSPEECHCOMMANDS(
            root="./data", subset="testing", sample_rate=SAMPLE_RATE, download=False
        )
    elif dataset == "timit":
        train = Timit(data_path="./data/timit", split="train", sample_rate=SAMPLE_RATE)
        test = Timit(data_path="./data/timit", split="test", sample_rate=SAMPLE_RATE)

    # return train, val, test # for SpeechCommands
    return train, test


def segment_dataset(
    audio_dataset,
    max_length=8000,
    segment_length=800,
    padding="constant",
    pad_value=0.0,
):
    """Segment long audio wavs into small segments of fixed length

    Args:
        audio_dataset (torch.utils.data.Dataset): A dataset that returns the tensors containing the
            wav content
        max_length (int): Maximum number of samples in a wav
        segment_length (int): Length of the each segment. Shorter segments will be padded to this
            length
        padding (str): Type of padding: [constant|gaussian|uniform]
        pad_value (float): Value for constant padding

    Yields:
        (torch.Tensor): The segmented audio data

    """
    assert (
        max_length % segment_length == 0
    ), f"max_length={max_length} should be divisible by ambient_dim={segment_length}"

    for data in tqdm(audio_dataset):
        if isinstance(data, tuple):
            wav = data[0]
        else:
            wav = data

        if wav.ndim > 1:  # channel reduction
            wav = wav.mean(0)
        wav = pad_tensor(wav, max_length, padding=padding, pad_value=pad_value)
        segmented = split_into_windows(
            wav, num_windows=int(max_length / segment_length)
        )

        for seg in segmented:
            if not seg.sum().item() == 0:  # Ignore all zero tensors (result of padding)
                yield seg


def measure_segments(
    segments_iterator,
    measurement_matrix,
    device="cpu",
):
    """Measure each segment using the measurement matrix

    Args:
        segments_iterator (Iterator): An iterator that yields the segmented audio tensors
        measurement_matrix (torch.Tensor): The measurement matrix
        device (str): Device to use for obtainign the measurements

    Yields:
        (Tuple[torch.Tensor, torch.Tensor]): (measurement, original) for each segment

    """
    def measure_x(x):
        # Create the measurements

        y = torch.einsum(
            "ma,ba->bm", measurement_matrix, x.squeeze().view(1, -1).to(device)
        ) 
        n = 1e-4 * torch.randn_like(y)
        y = y + n

        return y

    for segment in tqdm(segments_iterator):
        y = measure_x(segment)
        yield y.detach().cpu(), segment.detach().cpu()


def write_to_lmdb(iterator, db_path, write_frequency=5000):
    """Write segments to LMDB
    Segment data are too large to fit in system memory so dump them into lmdb for fast access

    Args:
        iterator (Iterator): Measurement iterator. Returns tuples (measurement, original)
        db_path (str): Path to dump the data
        write_frequency (int): Number of samples before flushing to disk
    """
    db = lmdb.open(
        db_path,
        subdir=False,
        map_size=1099511627776 * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)
    keys = []

    for idx, dat in enumerate(iterator):
        key = "{}".format(idx).encode("ascii")
        keys.append(key)
        byteflow = pickle.dumps(dat)
        txn.put(key, byteflow)

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()

    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))


def make_lmdb_data(
    dataset, db_path, measurement_matrix, device="cpu", write_frequency=5000
):
    """Wrapper for segmenting measuring and dumping to lmdb"""
    segment_iterator = segment_dataset(
        dataset,
        max_length=MAX_LENGTH,
        segment_length=AMBIENT_DIM,
        padding=PADDING,
    )
    measure_iterator = measure_segments(
        segment_iterator,
        measurement_matrix,
        device=device,
    )

    write_to_lmdb(measure_iterator, db_path, write_frequency=write_frequency)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, test = get_data_loaders(dataset=ARGS.dataset)

    out_folder = ARGS.out_folder

    if ARGS.out_folder is None:
        out_folder = (
            f"./data/{ARGS.dataset}_{NUM_MEASUREMENTS}_{AMBIENT_DIM}_{SAMPLE_RATE}_{NORMALIZE}/"
        )

    safe_mkdirs(out_folder)

    measurement_matrix = torch.randn(NUM_MEASUREMENTS, AMBIENT_DIM).to(device)

    if NORMALIZE == None:
        measurement_matrix = measurement_matrix
    elif NORMALIZE == "sqrt_m":
        measurement_matrix = measurement_matrix/np.sqrt(NUM_MEASUREMENTS)
    elif NORMALIZE == "orth":
        measurement_matrix = torch.nn.init.orthogonal_(measurement_matrix.t()).t().contiguous()
    else:
        print("Choose None, sqrt_m or orth")

    make_lmdb_data(
        train, os.path.join(out_folder, "train.lmdb"), measurement_matrix, device=device
    )
    # uncomment for the SpeechCommands dataset
    #make_lmdb_data(
    #    val, os.path.join(out_folder, "val.lmdb"), measurement_matrix, device=device
    #)
    make_lmdb_data(
        test, os.path.join(out_folder, "test.lmdb"), measurement_matrix, device=device
    )
    with open(os.path.join(out_folder, "measurement_matrix.p"), "wb") as fd:
        pickle.dump(measurement_matrix.detach().cpu(), fd)
