"""Dump clean measurements (without noise) to an existing lmdb"""

from torchvision import datasets
from datasets import MYSPEECHCOMMANDS, pad_tensor, split_into_windows, Timit
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import lmdb
import pickle
from tqdm import tqdm
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Segment and measure speech datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="speechcommands",
        help="which dataset to use [speechcommands|timit]",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=8000, help="Audio sample rate"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        default=None,
        help="Output folder to store the dataset",
    )

    return parser.parse_args()


ARGS = parse_args()

AMBIENT_DIM = 800
SAMPLE_RATE = ARGS.sample_rate
if ARGS.dataset == "speechcommands":
    MAX_LENGTH = SAMPLE_RATE  # Speech commands contains ~1sec segments
elif ARGS.dataset == "timit":
    MAX_LENGTH = int(
        60000 * (SAMPLE_RATE / 8000)
    )  # For sr == 8000 -> max length = 60000. SR=16000 -> MAX_LENGTH=120000

#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################

def get_data_loaders(dataset="speechcommands"):
    if dataset == "speechcommands":
        train = MYSPEECHCOMMANDS(
            root="./data",
            subset="training",
            sample_rate=SAMPLE_RATE,
            download=False,
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
    def measure_x(x):
        # Create y

        y = torch.einsum(
            "ma,ba->bm", measurement_matrix, x.squeeze().view(1, -1).to(device)
        )  # (400, 784) * (B,784) -> (B, 400)

        return y

    for segment in tqdm(segments_iterator):
        y = measure_x(segment)
        yield y.detach().cpu(), segment.detach().cpu()


def write_to_lmdb(iterator, db_path, write_frequency=5000):
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
    segment_iterator = segment_dataset(
        dataset,
        max_length=MAX_LENGTH,
        segment_length=AMBIENT_DIM,
        padding="constant",
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

    out_folder = ARGS.input_folder


    with open(f"{out_folder}/measurement_matrix.p", "rb") as fd:
        measurement_matrix = pickle.load(fd).to(device)

    make_lmdb_data(
        test, os.path.join(out_folder, "test.clean.lmdb"), measurement_matrix, device=device
    )
