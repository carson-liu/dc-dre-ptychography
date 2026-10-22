"""Checkpoint loading and batched inference."""

from pathlib import Path

import numpy as np
import torch

from models import LEGACY_NAMES, Calibration, build_model


def select_device():
    """Select CUDA, Apple MPS, or CPU in that order."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def disable_tf32():
    """Use exact float32 convolutions (cuDNN otherwise uses TF32)."""

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_checkpoint(checkpoint_path):
    """Safely load a state-dict checkpoint on CPU."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    name = checkpoint["representation"]
    checkpoint["representation"] = LEGACY_NAMES.get(name, name)
    return checkpoint


def build_model_from_checkpoint(checkpoint, device=None):
    """Build the network from an already loaded checkpoint dictionary."""

    device = torch.device(device) if device is not None else select_device()
    model = build_model(
        checkpoint["representation"],
        Calibration(**checkpoint["calibration"]),
        stack_views=checkpoint.get("stack_views", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_pretrained_model(checkpoint_path, device=None):
    """Load a checkpoint; returns the model and the checkpoint dictionary."""

    checkpoint = load_checkpoint(checkpoint_path)
    return build_model_from_checkpoint(checkpoint, device=device), checkpoint


def place_phase(phase, phase_range):
    """Place phases in the 2 pi interval centred on the label phase range.

    The branch cut then lies outside the data; for phase ranges inside
    (-pi, pi] this leaves the principal value unchanged.
    """

    low, high = phase_range
    start = low - (2 * np.pi - (high - low)) / 2
    return start + np.mod(phase - start, 2 * np.pi)


def reconstruct_patches(
    model, diffraction, amplitude_range, phase_range, batch_size=64, device=None
):
    """Reconstruct physical amplitude and phase for a stack of patterns.

    The network predicts the min-max normalised amplitude; it is mapped back
    with amplitude_range. The phase is atan2(s, c), placed with place_phase.
    """

    if diffraction.ndim != 3:
        raise ValueError("Expected diffraction with shape (N, height, width)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    amplitude_chunks = []
    phase_chunks = []

    with torch.inference_mode():
        for start in range(0, len(diffraction), batch_size):
            batch = torch.as_tensor(
                np.asarray(diffraction[start : start + batch_size]),
                dtype=torch.float32,
            ).unsqueeze(1)
            pred_amp, pred_cos, pred_sin = model(batch.to(device))
            pred_phase = torch.atan2(pred_sin, pred_cos)
            amplitude_chunks.append(pred_amp.squeeze(1).cpu().numpy())
            phase_chunks.append(pred_phase.squeeze(1).cpu().numpy())

    low, high = amplitude_range
    amplitude = low + np.concatenate(amplitude_chunks).astype(float) * (
        high - low
    )
    phase = place_phase(np.concatenate(phase_chunks).astype(float), phase_range)
    return amplitude, phase
