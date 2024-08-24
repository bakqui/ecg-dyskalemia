# Original work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2024 VUNO Inc. <minje.park@vuno.co>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import butter, resample, sosfiltfilt


__all__ = [
    'Resample',
    'RandomCrop',
    'CenterCrop',
    'NCrop',
    'ButterworthFilter',
    'Standardize',
    'ClassLabel',
    'ClassOneHot',
    'Compose',
    'ToTensor',
    'get_transforms_from_config',
]


"""Preprocessing1
"""
class Resample:
    """Resample the input sequence.
    """
    def __init__(
        self,
        target_length: Optional[int] = None,
        target_fs: Optional[int] = None,
    ) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig_len = x.shape[1]
        assert self.crop_length <= sig_len, \
            f"crop_length can't be larger than the length of x ({sig_len})."
        start_idx = np.random.randint(0, sig_len - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class CenterCrop:
    """Crop the input sequence at the center.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig_len = x.shape[1]
        assert self.crop_length <= sig_len, \
            f"crop_length can't be larger than the length of x ({sig_len})."
        start_idx = (x.shape[1] - self.crop_length) // 2
        return x[:, start_idx:start_idx + self.crop_length]

class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sig_len = x.shape[1]
        assert self.crop_length <= sig_len, \
            f"crop_length can't be larger than the length of x ({sig_len})."
        start_idx = np.arange(
            start=0,
            stop=sig_len - self.crop_length + 1,
            step=(sig_len - self.crop_length) // (self.num_segments - 1)
        )
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)

class ButterworthFilter:
    """Apply Butterworth filter to the input sequence.
    """
    def __init__(
        self,
        fs: int,
        cutoff: float,
        order: int = 5,
        btype: str = 'highpass',
    ) -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(
        self,
        axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2),
    ) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(
            x - loc,
            scale,
            out=np.zeros_like(x),
            where=scale != 0,
        )


"""Label transformation
"""
class ClassLabel:
    """Transform one-hot label to class label.
    """
    def __call__(self, y: np.ndarray) -> int:
        return np.argmax(y)

class ClassOneHot:
    """Transform class label to one-hot label.
    """
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, y: int) -> np.ndarray:
        return np.eye(self.num_classes)[y]


"""Etc
"""
class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


PREPROCESSING = {
    'resample': Resample,
    'random_crop': RandomCrop,
    'center_crop': CenterCrop,
    'n_crop': NCrop,
    'butter_filter': ButterworthFilter,
    'standardize': Standardize,
}


def get_transforms_from_config(
    config: Optional[List[Union[str, Dict[str, Any]]]] = None
) -> List[callable]:
    """Get transforms from config.
    """
    config = config or []
    transforms = []
    for transform in config:
        if isinstance(transform, str):
            name = transform
            kwargs = {}
        elif isinstance(transform, dict):
            assert len(transform) == 1, \
                "Each transform must have only one key."
            name, kwargs = list(transform.items())[0]
        else:
            raise ValueError(
                f"Invalid transform: {transform}, it must be a str or a dict."
            )
        if name in PREPROCESSING:
            transforms.append(PREPROCESSING[name](**kwargs))
        else:
            raise ValueError(f"Invalid name: {name}")
    return transforms
