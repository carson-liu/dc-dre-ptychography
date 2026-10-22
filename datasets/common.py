"""Shared containers and split helpers for the data pipelines."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, random_split

from models import Calibration, calibrate


@dataclass(frozen=True)
class ReconstructionTestSet:
    """Diffraction inputs and reconstruction targets for one test region."""

    diffraction: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    grid_size: int


@dataclass(frozen=True)
class PreparedData:
    """Training/validation datasets plus dataset-specific metadata."""

    train_dataset: Dataset
    validation_dataset: Dataset
    test_set: ReconstructionTestSet
    calibration: Calibration
    amplitude_range: tuple
    phase_range: tuple


def split_training_data(
    diffraction,
    amplitude,
    phase,
    train_mask,
    validation_size,
    seed=0,
    calibration_subsample=1,
):
    """Shuffle the training positions, calibrate, and split off validation.

    Amplitude targets are min-max normalised with the dataset range. The
    calibration uses all measured patterns for the low gain and the training
    patterns for the disc radius and the high gain.
    """

    patch_shape = diffraction.shape[-2:]
    amplitude_range = (float(amplitude.min()), float(amplitude.max()))
    phase_range = (float(phase.min()), float(phase.max()))
    low, high = amplitude_range

    x_train = diffraction[train_mask].reshape(-1, 1, *patch_shape)
    amp_train = ((amplitude[train_mask] - low) / (high - low)).astype(
        np.float32
    )
    amp_train = amp_train.reshape(-1, 1, *patch_shape)
    phase_train = phase[train_mask].reshape(-1, 1, *patch_shape)
    permutation = np.random.RandomState(0).permutation(len(x_train))
    x_train = x_train[permutation]
    amp_train = amp_train[permutation]
    phase_train = phase_train[permutation]

    calibration = calibrate(
        diffraction, x_train[::calibration_subsample, 0]
    )
    dataset = TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(amp_train, dtype=torch.float32),
        torch.as_tensor(phase_train, dtype=torch.float32),
    )
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError(f"Invalid validation size: {validation_size}")
    train_dataset, validation_dataset = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        train_dataset,
        validation_dataset,
        calibration,
        amplitude_range,
        phase_range,
    )
