"""Experimental X-ray dataset using the original measurement values."""

from pathlib import Path

import numpy as np

from .common import PreparedData, ReconstructionTestSet, split_training_data


EXPERIMENTAL_DIFFRACTION_FILE = "20191008_39_diff_reduced.npz"
EXPERIMENTAL_OBJECT_FILE = "20191008_39_amp_pha_10nm_full.npy"


def _check_paths(root):
    root = Path(root)
    diffraction_path = root / EXPERIMENTAL_DIFFRACTION_FILE
    object_path = root / EXPERIMENTAL_OBJECT_FILE
    missing = [
        path.name for path in (diffraction_path, object_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing experimental data files: {missing}")
    return diffraction_path, object_path


def _extract_experimental_test_set(diffraction, complex_object, grid_size):
    test_start = complex_object.shape[0] - grid_size
    patch_shape = diffraction.shape[-2:]
    test_diffraction = diffraction[test_start:, :grid_size].reshape(
        -1, *patch_shape
    )
    test_object = complex_object[test_start:, :grid_size]
    test_amplitude = np.abs(test_object).reshape(-1, *patch_shape)
    test_phase = np.angle(test_object).reshape(-1, *patch_shape)
    return ReconstructionTestSet(
        diffraction=test_diffraction.astype(np.float32, copy=False),
        amplitude=test_amplitude.astype(np.float32, copy=False),
        phase=test_phase.astype(np.float32, copy=False),
        grid_size=grid_size,
    )


def prepare_experimental_data(
    root=".",
    grid_size=60,
    validation_size=805,
    seed=0,
    calibration_subsample=7,
    train_rows=101,
):
    """Prepare the experimental training, validation, and test splits.

    Test: the last 60 scan rows x first 60 columns (3600 patterns).
    Training and validation pool: the first 101 scan rows (16261 patterns,
    805 of them for validation); the remaining 6060 patterns are not used.
    The disc radius and high gain are calibrated on a fixed one-in-seven
    subsample of the training patterns.
    """

    diffraction_path, object_path = _check_paths(root)
    with np.load(diffraction_path) as archive:
        diffraction = archive["arr_0"].astype(np.float32)
    complex_object = np.load(object_path)
    amplitude = np.abs(complex_object).astype(np.float32)
    phase = np.angle(complex_object).astype(np.float32)

    scan_shape = amplitude.shape[:2]
    test_start = scan_shape[0] - grid_size
    mask = np.zeros(scan_shape, dtype=bool)
    mask[:train_rows] = True
    mask[test_start:, :grid_size] = False

    train, validation, calibration, amp_range, phase_range = (
        split_training_data(
            diffraction,
            amplitude,
            phase,
            mask,
            validation_size,
            seed=seed,
            calibration_subsample=calibration_subsample,
        )
    )
    return PreparedData(
        train_dataset=train,
        validation_dataset=validation,
        test_set=_extract_experimental_test_set(
            diffraction, complex_object, grid_size
        ),
        calibration=calibration,
        amplitude_range=amp_range,
        phase_range=phase_range,
    )


def load_experimental_test_set(root=".", grid_size=60):
    """Load the experimental test region without data normalization."""

    diffraction_path, object_path = _check_paths(root)
    with np.load(diffraction_path) as archive:
        diffraction = archive["arr_0"]
    complex_object = np.load(object_path, mmap_mode="r")
    return _extract_experimental_test_set(
        diffraction, complex_object, grid_size
    )
