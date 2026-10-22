"""Numerical representations of diffraction intensities.

A representation maps a measured intensity pattern I (detector counts) to one
or more parameter-free views that the encoders receive:

    raw        I
    diff_max   I / max(I) over the dataset
    linear     I_l = min(g_l I, I_sat)
    high_only  I_h = min(g_h I, I_sat)
    sqrt       I_sat sqrt(I_l / I_sat)
    log        I_sat log(1 + beta I_l / I_sat) / log(1 + beta)
    dc_dre     (I_l, I_h): Diffraction-Calibrated Dual-Range Encoding   (default)
    masked     (I_l, min(g_h I 1[|q| > r_d], I_sat))

The gains are calibrated once per dataset from diffraction data only.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn


SATURATION = 4095.0  # range of a 12-bit detector

# name -> (number of views, single encoder widened to the two-encoder size)
REPRESENTATIONS = {
    "raw": (1, True),
    "diff_max": (1, True),
    "linear": (1, True),
    "high_only": (1, True),
    "sqrt": (1, True),
    "log": (1, True),
    "dc_dre": (2, False),
    "masked": (2, False),
}
# names used by earlier checkpoints
LEGACY_NAMES = {"dual_gain": "dc_dre", "masked_high_gain": "masked"}


@dataclass(frozen=True)
class Calibration:
    """Gains of DC-DRE, computed once per dataset."""

    low_gain: float
    high_gain: float
    disc_radius: float
    diffraction_max: float

    @property
    def log_beta(self):
        return solve_log_beta(self.high_gain / self.low_gain)

    def to_dict(self):
        return asdict(self)


def disc_radius(patterns, energy=0.9):
    """Smallest radius that encloses `energy` of the mean pattern."""

    mean = patterns.mean(0)
    y, x = np.indices(mean.shape)
    radius_map = np.hypot(y - mean.shape[0] // 2, x - mean.shape[1] // 2)
    for radius in np.arange(0.5, mean.shape[0] / 2, 0.5):
        if mean[radius_map <= radius].sum() >= energy * mean.sum():
            return float(radius)
    return float(mean.shape[0] / 2)


def calibrate(
    all_patterns, train_patterns, headroom=0.85, percentile=99.5, energy=0.9
):
    """Calibrate the representation from diffraction data alone.

    g_l maps the 99.5th percentile of all measured intensities to 85% of the
    detector range. r_d encloses 90% of the energy of the mean training
    pattern, and g_h maps the 99.5th percentile of the training intensities
    outside this disc to 85% of the range. No labels or probe are used.
    """

    all_patterns = np.asarray(all_patterns, dtype=np.float32)
    train_patterns = np.asarray(train_patterns, dtype=np.float32)
    height, width = train_patterns.shape[-2:]
    radius = disc_radius(train_patterns, energy)
    y, x = np.indices((height, width))
    outside = np.hypot(y - height // 2, x - width // 2) > radius
    low_gain = headroom * SATURATION / np.percentile(all_patterns, percentile)
    high_gain = headroom * SATURATION / np.percentile(
        train_patterns * outside, percentile
    )
    return Calibration(
        low_gain=float(low_gain),
        high_gain=float(high_gain),
        disc_radius=radius,
        diffraction_max=float(all_patterns.max()),
    )


def solve_log_beta(ratio):
    """Return beta with beta / log(1 + beta) = ratio (slope at zero)."""

    low, high = 0.0, 1.0
    while high / math.log1p(high) < ratio:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if middle / math.log1p(middle) < ratio:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class InputRepresentation(nn.Module):
    """Parameter-free input representation returning a tuple of views."""

    def __init__(self, name, calibration):
        super().__init__()
        name = LEGACY_NAMES.get(name, name)
        if name not in REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation {name!r}; "
                f"choose from {list(REPRESENTATIONS)}"
            )
        self.name = name
        self.calibration = calibration
        self._mask = None

    @property
    def num_views(self):
        return REPRESENTATIONS[self.name][0]

    @property
    def widened(self):
        return REPRESENTATIONS[self.name][1]

    def _disc_mask(self, intensity):
        height, width = intensity.shape[-2:]
        if (
            self._mask is None
            or self._mask.shape[-2:] != (height, width)
            or self._mask.device != intensity.device
        ):
            y, x = np.indices((height, width))
            outside = np.hypot(y - height // 2, x - width // 2)
            mask = outside > self.calibration.disc_radius
            self._mask = torch.tensor(
                mask.astype(np.float32), device=intensity.device
            )[None, None]
        return self._mask

    def forward(self, intensity):
        calibration = self.calibration
        if self.name == "raw":
            return (intensity,)
        if self.name == "diff_max":
            return (intensity / calibration.diffraction_max,)
        low = (calibration.low_gain * intensity).clamp(0, SATURATION)
        if self.name == "linear":
            return (low,)
        if self.name == "sqrt":
            return (SATURATION * torch.sqrt(low / SATURATION),)
        if self.name == "high_only":
            return ((calibration.high_gain * intensity).clamp(0, SATURATION),)
        if self.name == "log":
            beta = calibration.log_beta
            log_view = SATURATION * torch.log1p(beta * low / SATURATION)
            return (log_view / math.log1p(beta),)
        if self.name == "dc_dre":
            high = (calibration.high_gain * intensity).clamp(0, SATURATION)
            return low, high
        high = calibration.high_gain * intensity * self._disc_mask(intensity)
        return low, high.clamp(0, SATURATION)

    def as_channels(self, intensity):
        """Views stacked as channels; gain-based views scaled to [0, 1].

        This is a drop-in input for other reconstruction networks.
        """

        views = torch.cat(self(intensity), dim=1)
        if self.name in ("raw", "diff_max"):
            return views
        return views / SATURATION
