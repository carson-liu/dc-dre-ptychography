"""Simulated AD_LTEM dataset (noisy and noise-free)."""

from pathlib import Path

import numpy as np

from .common import PreparedData, ReconstructionTestSet, split_training_data


AD_LTEM_FILES = {
    "amplitude": "75_amp.npy",
    "phase": "75_ph.npy",
    "diffraction": "75_diff.npy",
    "diffraction_noise_free": "75_diff_noise_free.npy",
}


def _load_arrays(root, noise_free=False):
    root = Path(root)
    names = ["amplitude", "phase"]
    names.append("diffraction_noise_free" if noise_free else "diffraction")
    paths = {name: root / AD_LTEM_FILES[name] for name in names}
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing AD_LTEM data files: {missing}. "
            "Run `python main.py --mode generate` first."
        )
    arrays = {name: np.load(path) for name, path in paths.items()}
    arrays["diffraction"] = arrays.pop(names[-1]).astype(np.float32)
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("AD_LTEM diffraction, amplitude, and phase shapes differ")
    return arrays


def _extract_test_set(arrays, grid_size):
    patch_shape = arrays["diffraction"].shape[-2:]

    def extract(array):
        return (
            array[-grid_size:, -grid_size:]
            .reshape(-1, *patch_shape)
            .astype(np.float32, copy=False)
        )

    return ReconstructionTestSet(
        diffraction=extract(arrays["diffraction"]),
        amplitude=extract(arrays["amplitude"]),
        phase=extract(arrays["phase"]),
        grid_size=grid_size,
    )


def load_ad_ltem_test_set(root=".", grid_size=12, noise_free=False):
    """Load the lower-right 12 x 12 AD_LTEM test region used in the paper."""

    return _extract_test_set(_load_arrays(root, noise_free), grid_size)


def prepare_ad_ltem_data(
    root=".",
    grid_size=12,
    validation_size=100,
    seed=0,
    noise_free=False,
    calibration_subsample=1,
    train_rows=49,
):
    """Prepare the AD_LTEM training, validation, and test splits.

    Test: the last grid_size x grid_size scan positions (144 patterns).
    Training and validation pool: the first 49 scan rows (2989 patterns,
    100 of them for validation); the remaining 588 patterns are not used.
    """

    arrays = _load_arrays(root, noise_free)
    scan_shape = arrays["amplitude"].shape[:2]
    test_start = scan_shape[0] - grid_size
    mask = np.zeros(scan_shape, dtype=bool)
    mask[:train_rows] = True
    mask[test_start:, test_start:] = False

    train, validation, calibration, amp_range, phase_range = (
        split_training_data(
            arrays["diffraction"],
            arrays["amplitude"],
            arrays["phase"],
            mask,
            validation_size,
            seed=seed,
            calibration_subsample=calibration_subsample,
        )
    )
    return PreparedData(
        train_dataset=train,
        validation_dataset=validation,
        test_set=_extract_test_set(arrays, grid_size),
        calibration=calibration,
        amplitude_range=amp_range,
        phase_range=phase_range,
    )
