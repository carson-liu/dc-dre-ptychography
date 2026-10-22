"""Shared training loop for AD_LTEM and experimental data."""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ReconstructionLoss, build_model

from .inference import disable_tf32, select_device


def _run_epoch(model, data_loader, loss_function, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    description = "Train" if training else "Validation"

    for diffraction, amplitude, phase in tqdm(
        data_loader, desc=description, leave=False
    ):
        diffraction = diffraction.to(device)
        amplitude = amplitude.to(device)
        phase = phase.to(device)

        with torch.set_grad_enabled(training):
            loss = loss_function(model(diffraction), amplitude, phase)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
        total += loss.item()

    return total / len(data_loader)


def train_model(
    prepared_data,
    dataset_name,
    checkpoint_path,
    representation="dc_dre",
    stack_views=False,
    epochs=40,
    batch_size=64,
    learning_rate=1e-3,
    seed=0,
    device=None,
):
    """Train the network with the protocol of the paper.

    Adam with a triangular-2 cyclic learning rate between learning_rate / 10
    and learning_rate (half cycle of six epochs), gradient-norm clipping at 1,
    and the model after the last epoch.
    """

    device = torch.device(device) if device is not None else select_device()
    disable_tf32()
    torch.manual_seed(seed)
    train_loader = DataLoader(
        prepared_data.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        prepared_data.validation_dataset, batch_size=batch_size
    )

    model = build_model(
        representation, prepared_data.calibration, stack_views=stack_views
    ).to(device)
    loss_function = ReconstructionLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterations_per_epoch = math.floor(
        len(prepared_data.train_dataset) / batch_size
    ) + 1
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=learning_rate / 10,
        max_lr=learning_rate,
        step_size_up=6 * iterations_per_epoch,
        cycle_momentum=False,
        mode="triangular2",
    )

    calibration = prepared_data.calibration
    parameters = sum(p.numel() for p in model.parameters())
    print(
        f"Dataset={dataset_name} | representation={representation}"
        f"{' (stacked views)' if stack_views else ''} | "
        f"parameters={parameters / 1e6:.2f} M | seed={seed} | device={device}"
    )
    print(
        f"Calibration: g_l={calibration.low_gain:.4g} | "
        f"g_h={calibration.high_gain:.4g} | "
        f"g_h/g_l={calibration.high_gain / calibration.low_gain:.2f} | "
        f"r_d={calibration.disc_radius} px"
    )
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        validation_loss = _run_epoch(
            model, validation_loader, loss_function, device
        )
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{epochs} | {elapsed:.1f}s | lr={current_lr:.6f} | "
            f"train={train_loss:.5f} | validation={validation_loss:.5f}"
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "representation": representation,
            "stack_views": stack_views,
            "calibration": calibration.to_dict(),
            "amplitude_range": list(prepared_data.amplitude_range),
            "phase_range": list(prepared_data.phase_range),
            "input_size": prepared_data.test_set.diffraction.shape[-1],
            "dataset": dataset_name,
            "epoch": epochs,
            "seed": seed,
        },
        checkpoint_path,
    )
    print(f"Checkpoint (last epoch): {checkpoint_path}")
    return model.eval()
