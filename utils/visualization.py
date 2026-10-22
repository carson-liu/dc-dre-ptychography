"""Input-representation, patch, and stitched-result visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import InputRepresentation

from .metrics import format_metrics, normalized_patch_metrics


def _save_figure(figure, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")


def plot_input_representations(pattern, calibration, crop=10, save_path=None):
    """Show the encoder views of one diffraction pattern.

    crop: half-width of the central region shown (None for the full pattern).
    """

    intensity = torch.as_tensor(np.asarray(pattern), dtype=torch.float32)
    intensity = intensity[None, None]

    def view(name, index=0):
        return InputRepresentation(name, calibration)(intensity)[index][0, 0]

    panels = [
        ("Raw counts (log10)", np.log10(np.asarray(pattern, dtype=float) + 1)),
        ("Low gain $I_l$ (linear)", view("linear").numpy()),
        ("Log", view("log").numpy()),
        ("High gain $I_h$", view("dc_dre", 1).numpy()),
        ("Masked high gain", view("masked", 1).numpy()),
    ]
    figure, axes = plt.subplots(
        1, len(panels), figsize=(3.2 * len(panels), 3.2), constrained_layout=True
    )
    for axis, (label, image) in zip(axes, panels):
        if crop is not None:
            center = image.shape[0] // 2
            image = image[center - crop : center + crop, center - crop : center + crop]
        shown = axis.imshow(image, cmap="magma")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.02)
        axis.set_title(label, fontsize=11)
        axis.axis("off")
    _save_figure(figure, save_path)
    return figure


def plot_patch_comparison(
    diffraction,
    amp_true,
    amp_pred,
    phase_true,
    phase_pred,
    amplitude_range,
    phase_range,
    indices=(1, 3, 79),
    grid_size=12,
    save_path=None,
):
    """Plot patch comparisons and return their individual metrics.

    The metrics are the canonical per-patch metrics (utils.metrics), computed
    over the whole test set and reported for the selected patches.
    """

    column_titles = [
        "Diffraction (log10)",
        "Amplitude GT",
        "Amplitude prediction",
        "Phase GT",
        "Phase prediction",
    ]
    figure, axes = plt.subplots(
        len(indices), 5, figsize=(13.5, 3 * len(indices)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    amp_all = normalized_patch_metrics(amp_true, amp_pred, amplitude_range)
    phase_all = normalized_patch_metrics(phase_true, phase_pred, phase_range)
    patch_metrics = []
    for row, index in enumerate(indices):
        amp_metrics, phase_metrics = amp_all[index], phase_all[index]
        scan_row, scan_column = divmod(index, grid_size)
        patch_metrics.append(
            {
                "index": index,
                "scan_position": (scan_row, scan_column),
                "amplitude": amp_metrics,
                "phase": phase_metrics,
            }
        )

        images = [
            np.log10(diffraction[index] + 1),
            amp_true[index],
            amp_pred[index],
            phase_true[index],
            phase_pred[index],
        ]
        limits = [
            None,
            (amp_true[index].min(), amp_true[index].max()),
            (amp_true[index].min(), amp_true[index].max()),
            (phase_true[index].min(), phase_true[index].max()),
            (phase_true[index].min(), phase_true[index].max()),
        ]
        for column, (image, limit) in enumerate(zip(images, limits)):
            vmin, vmax = limit if limit is not None else (None, None)
            axes[row, column].imshow(image, vmin=vmin, vmax=vmax)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(column_titles[column], fontsize=11)

        axes[row, 0].text(
            -0.08,
            0.5,
            f"Patch {index}\n({scan_row}, {scan_column})",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        for column, metrics in ((2, amp_metrics), (4, phase_metrics)):
            axes[row, column].text(
                0.5,
                -0.08,
                format_metrics(metrics, multiline=True),
                transform=axes[row, column].transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )

    figure.suptitle("Patch reconstruction", fontsize=15)
    _save_figure(figure, save_path)
    return figure, patch_metrics


def plot_global_stitching(
    amp_true,
    amp_pred,
    phase_true,
    phase_pred,
    amp_metrics,
    phase_metrics,
    save_path=None,
):
    """Plot stitched ground truth and reconstruction with stitched metrics."""

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    panels = [
        (axes[0, 0], amp_true, "Amplitude GT", amp_true),
        (
            axes[0, 1],
            amp_pred,
            f"Amplitude prediction\n{format_metrics(amp_metrics, multiline=True)}",
            amp_true,
        ),
        (axes[1, 0], phase_true, "Phase GT", phase_true),
        (
            axes[1, 1],
            phase_pred,
            f"Phase prediction\n{format_metrics(phase_metrics, multiline=True)}",
            phase_true,
        ),
    ]
    for axis, image, title, reference in panels:
        axis.imshow(image, vmin=reference.min(), vmax=reference.max())
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle("Stitched reconstruction of the test region", fontsize=15)
    _save_figure(figure, save_path)
    return figure
