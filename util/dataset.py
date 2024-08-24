# Original work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2024 VUNO Inc. <minje.park@vuno.co>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle as pkl
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import util.transforms as T
from util.transforms import get_transforms_from_config
from util.misc import get_rank, get_world_size


def load_ndarray(fpath: str) -> np.ndarray:
    """Load 12-lead np.ndarray data saved as .npy file.
    """
    x = np.load(fpath)
    return x


def load_pkl(fpath: str) -> np.ndarray:
    """Load 12-lead np.ndarray data saved as .pkl file.
    """
    with open(fpath, 'rb') as f:
        x = pkl.load(f)
    assert isinstance(x, np.ndarray), f"Data should be numpy array. ({fpath})"
    return x


def load_ecg_dict(fpath: str) -> np.ndarray:
    """Load 12-lead ECG data saved as Dict[str, np.ndarray].
    """
    _12lead_keys = [
        "I", "II", "III", "AVR", "AVL", "AVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    with open(fpath, 'rb') as f:
        ecg_dict = pkl.load(f)
    assert isinstance(ecg_dict, dict), f"Data should be dictionary. ({fpath})"
    assert set(_12lead_keys).issubset(ecg_dict.keys()), \
        f"12-lead ECG keys are missing. ({fpath})"
    x = np.stack([ecg_dict[k] for k in _12lead_keys], axis=0)
    return x


class ECGDataset(Dataset):
    _LEAD_SLICE = {
        "12lead": slice(0, 12),
        "limb_lead": slice(0, 6),
        "lead1": slice(0, 1),
    }

    def __init__(
        self,
        root_dir: str,
        filenames: Iterable = None,
        labels: Optional[Iterable] = None,
        fs_list: Optional[Iterable] = None,
        target_lead: Literal["12lead", "limb_lead", "lead1"] = "12lead",
        target_fs: int = 250,
        transform: Optional[object] = None,
        label_transform: Optional[object] = None,
        loader: callable = load_ecg_dict,
        extension: str = ".pkl",
    ):
        """
        Args:
            root_dir: Directory with all the data.
            filenames: List of filenames. (.pkl)
            fs_list: List of sampling rates.
            labels: List of labels.
            target_lead: lead to use. {'limb_lead', 'lead1', 'lead2'}
            target_fs: Target sampling rate.
            transform: Optional transform to be applied on a sample.
            label_transform: Optional transform to be applied on a label.
            loader: Function to load data. (default: load_ecg_dict)
            extension: File extension. (default: ".pkl")
        """
        self.root_dir = root_dir
        self.loader = loader
        self.extension = extension
        self.filenames = filenames
        self.labels = labels
        self.fs_list = fs_list
        self.target_lead = target_lead
        self.target_fs = target_fs
        self.check_dataset()
        self.resample = T.Resample(target_fs=target_fs) if fs_list is not None else None

        self.transform = transform
        self.label_transform = label_transform

    def check_dataset(self):
        fname_invalid_ext = [
            f for f in self.filenames if not f.endswith(self.extension)
        ]
        assert len(fname_invalid_ext) == 0, \
            f"Some files have invalid extension. ({fname_invalid_ext[0]})"
        fpaths = [
            os.path.join(self.root_dir, fname) for fname in self.filenames
        ]
        assert all([os.path.exists(fpath) for fpath in fpaths]), \
            f"Some files do not exist. ({fpaths[0]})"
        if self.fs_list is not None:
            assert len(self.filenames) == len(self.fs_list), \
                "The number of filenames and fs_list are different."
        if self.labels is not None:
            assert len(self.filenames) == len(self.labels), \
                "The number of filenames and labels are different."
        assert self.target_lead in self._LEAD_SLICE.keys(), \
            f"target_lead should be one of {list(self._LEAD_SLICE.keys())}"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        fpath = os.path.join(self.root_dir, fname)
        x = self.loader(fpath)
        x = x[self._LEAD_SLICE[self.target_lead]]
        if self.resample is not None:
            x = self.resample(x, self.fs_list[idx])
        if self.transform:
            x = self.transform(x)

        if self.labels is not None:
            y = self.labels[idx]
            if self.label_transform:
                y = self.label_transform(y)
            return x, y
        else:
            return x


def build_dataset(dataset_cfg: dict, split: str) -> ECGDataset:
    """
    Load train, validation, and test dataloaders.
    """
    index_dir = os.path.realpath(dataset_cfg["index_dir"])
    ecg_dir = os.path.realpath(dataset_cfg["ecg_dir"])

    fname_col = dataset_cfg.get("filename_col", "FILE_NAME")
    fs_col = dataset_cfg.get("fs_col", None)
    label_col = dataset_cfg.get("label_col", None)
    df_name = dataset_cfg.get(f"{split}_index", None)
    assert df_name is not None, f"{split}_index is not defined in the config."
    if df_name.endswith(".csv"):
        df = pd.read_csv(os.path.join(index_dir, df_name))
    elif df_name.endswith(".pkl"):
        df = pd.read_pickle(os.path.join(index_dir, df_name))
    filenames = df[fname_col].tolist()
    fs_list = df[fs_col].astype(int).tolist() if fs_col is not None else None
    labels = df[label_col].values if label_col is not None else None

    target_lead = dataset_cfg.get("lead", "12lead")
    target_fs = dataset_cfg.get("target_fs", 250)

    if split == "train":
        transforms = get_transforms_from_config(dataset_cfg["train_transforms"])
    else:
        transforms = get_transforms_from_config(dataset_cfg["eval_transforms"])
    transforms = T.Compose(transforms + [T.ToTensor()])
    if labels is not None:
        label_transforms = get_transforms_from_config(
            dataset_cfg.get("label_transforms", [])
        )
        label_transform = T.Compose(
            label_transforms + [T.ToTensor(dtype=dataset_cfg["label_dtype"])]
        )
    else:
        label_transform = None

    loader_name = dataset_cfg.get("loader", "load_ecg_dict")
    loader = globals().get(loader_name, load_ecg_dict)
    extension = dataset_cfg.get("extension", ".pkl")

    dataset = ECGDataset(
        ecg_dir,
        filenames=filenames,
        fs_list=fs_list,
        labels=labels,
        target_lead=target_lead,
        target_fs=target_fs,
        transform=transforms,
        label_transform=label_transform,
        loader=loader,
        extension=extension,
    )

    return dataset


def get_dataloader(
    dataset: Dataset,
    is_distributed: bool = False,
    dist_eval: bool = False,
    mode: Literal["train", "eval"] = "train",
    **kwargs
) -> DataLoader:
    is_train = mode == "train"
    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.'
            )
        # shuffle=True to reduce monitor bias even if it is for validation.
        # https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L189
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True,
        )
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=is_train,
        **kwargs
    )
