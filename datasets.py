# from admm_cifar import safe_mkdirs
from torch.utils import data
from torch.utils.data import Dataset
import torchaudio
import glob
import os
import torch
from typing import List, Tuple, Union, Optional
from torchaudio.datasets import SPEECHCOMMANDS
from pathlib import Path
import pickle
from tqdm import tqdm
import lmdb

Label = Union[torch.Tensor, int]
Device = Union[torch.device, str]


def mktensor(
    data: torch.Tensor,
    dtype: torch.dtype = torch.float,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def pad_tensor(
    t: torch.Tensor, pad_length, padding="constant", pad_value=0.0
) -> torch.Tensor:
    trailing_dims = t.size()[1:]
    dims = (pad_length,) + trailing_dims
    if padding == "constant":
        out_t = t.new_full(dims, pad_value)
    elif padding == "gaussian":
        out_t = torch.randn(dims).to(t.device).type(t.type)
    elif padding == "uniform":
        out_t = torch.rand(dims).to(t.device).type(t.type)
    else:
        raise ValueError(
            f"padding should be one of [constant,gaussian,uniform]. {padding} given"
        )
    out_t[: min(t.size(0), pad_length), ...] = t[: min(t.size(0), pad_length), ...]
    return out_t


def split_into_windows(t: torch.Tensor, num_windows: int = 10) -> torch.Tensor:
    return [x for x in t.reshape(num_windows, -1)]


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: Union[float, int] = 0.0,
    max_length: int = -1,
):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_length < 0:
        max_len = max([s.size(0) for s in sequences])
    else:
        max_len = max_length
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, : min(length, max_len), ...] = tensor[: min(length, max_len), ...]

    return out_tensor


class SequenceCollator(object):
    def __init__(self, pad_indx=0, max_length=-1, device="cpu"):
        self.pad_indx = pad_indx
        self.device = device
        self.max_length = max_length

    def __call__(
        self, batch: List[Tuple[torch.Tensor, Label]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call collate function
        Args:
            batch (List[Tuple[torch.Tensor, slp.util.types.Label]]): Batch of samples.
                It expects a list of tuples (inputs, label).
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns tuple of batched tensors (inputs, labels, lengths)
        """
        inputs: List[torch.Tensor] = [b[0] for b in batch]
        targets: List[Label] = [b[1] for b in batch]
        #  targets: List[torch.tensor] = map(list, zip(*batch))
        lengths = torch.tensor([s.size(0) for s in inputs], device=self.device)

        if self.max_length > 0:
            lengths = torch.clamp(lengths, min=0, max=self.max_length)
        # Pad and convert to tensor
        inputs_padded: torch.Tensor = pad_sequence(
            inputs,
            padding_value=self.pad_indx,
            max_length=self.max_length,
        ).to(self.device)

        ttargets: torch.Tensor = mktensor(targets, device=self.device, dtype=torch.long)

        return inputs_padded, ttargets.to(self.device)


def apply_audio_transforms(wav, sr, target_sr=8000):
    wav = wav.mean(dim=0)  # Average channels together
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)

    return wav


class Timit(Dataset):
    """Timit dataloader
    Assumes timit wavs are flattened according to
    flatten_timit.py script for simplicity
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        lazy: bool = False,
        sample_rate: int = 8000,
        padding: str = "constant",
        max_length: int = -1,
    ) -> None:
        self.padding = padding
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.lazy = lazy
        self.paths = glob.glob(os.path.join(data_path, split, "*.wav"))
        if lazy:
            self.data = self.paths
        else:
            self.data = [
                apply_audio_transforms(
                    *torchaudio.load(p, normalize=True), target_sr=sample_rate
                )
                for p in self.paths
            ]

        self.gender = [p.split("_")[1][0] for p in self.paths]
        self.dialect = [p.split("_")[0] for p in self.paths]
        self.speaker = [p.split("_")[1] for p in self.paths]
        self.utterance = [p.split(".")[0].split("_")[-1] for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        wav = self.data[index]
        if self.lazy:
            wav = apply_audio_transforms(
                *torchaudio.load(wav, normalize=True), target_sr=self.sample_rate
            )
        if self.max_length > 0:
            wav = pad_tensor(wav, self.max_length, padding=self.padding)
        return wav, 0  # Return dummy label to plug in our code out of the box


class Audioset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        lazy: bool = True,
        sample_rate: int = 8000,
    ) -> None:
        self.lazy = lazy
        self.paths = glob.glob(os.path.join(data_path, split, "*.wav"))
        if lazy:
            self.data = self.paths
        else:
            # Will probably run out of mem in the home pc. Just do lazy load
            self.data = [
                apply_audio_transforms(
                    *torchaudio.load(p, normalize=True), target_sr=sample_rate
                )
                for p in self.paths
            ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        wav = self.data[index]
        if self.lazy:
            wav = apply_audio_transforms(
                *torchaudio.load(wav, normalize=True), target_sr=self.sample_rate
            )

        return wav, 0  # Return dummy label to plug in our code out of the box


class MYSPEECHCOMMANDS(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        url: str = "speech_commands_v0.02",
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: Optional[str] = "training",
        sample_rate: int = 8000,
        max_length: int = -1,
        padding: str = "constant",
    ) -> None:
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.padding = padding
        super().__init__(
            root,
            url=url,
            folder_in_archive=folder_in_archive,
            download=download,
            subset=subset,
        )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, str]:
        wav, sr, _, _, _ = super().__getitem__(n)
        wav = apply_audio_transforms(wav, sr, target_sr=self.sample_rate)
        if self.max_length > 0:
            wav = pad_tensor(wav, self.max_length, padding=self.padding)
        return wav, 0


class MeasurementsDataset(Dataset):
    def __init__(self, measurements_data_folder, split="train"):
        db_path = os.path.join(measurements_data_folder, f"{split}.lmdb")
        self.env = lmdb.open(
            db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        y, x = pickle.loads(byteflow)
        return y, x


if __name__ == "__main__":
    data = Timit("./data/timit", split="train")
    import ipdb

    ipdb.set_trace()
