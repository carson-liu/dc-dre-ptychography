"""Train, evaluate, or generate data for dynamic-range-aware reconstruction."""

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    load_ad_ltem_test_set,
    load_experimental_test_set,
    prepare_ad_ltem_data,
    prepare_experimental_data,
)
from models import REPRESENTATIONS
from simulation import generate_ad_ltem_dataset
from utils import (
    AdaptiveImageStitcher,
    ExperimentalImageStitcher,
    disable_tf32,
    evaluate_reconstruction,
    format_average_metrics,
    format_metrics,
    load_pretrained_model,
    plot_global_stitching,
    plot_patch_comparison,
    reconstruct_patches,
    select_device,
    train_model,
)


REPO_ROOT = Path(__file__).resolve().parent
SIMULATION_ASSETS = REPO_ROOT / "simulation" / "assets"

DATASETS = {
    "ad_ltem": {
        "prepare": partial(prepare_ad_ltem_data, noise_free=False),
        "load_test": partial(load_ad_ltem_test_set, noise_free=False),
        "grid_size": 12,
        "validation_size": 100,
        "epochs": 40,
        "patch_indices": (1, 3, 79),
        "stitcher": lambda: AdaptiveImageStitcher(point_size=8, overlap=5 * 8),
        "normalize": True,
    },
    "ad_ltem_noise_free": {
        "prepare": partial(prepare_ad_ltem_data, noise_free=True),
        "load_test": partial(load_ad_ltem_test_set, noise_free=True),
        "grid_size": 12,
        "validation_size": 100,
        "epochs": 40,
        "patch_indices": (1, 3, 79),
        "stitcher": lambda: AdaptiveImageStitcher(point_size=8, overlap=5 * 8),
        "normalize": True,
    },
    "experimental": {
        "prepare": prepare_experimental_data,
        "load_test": load_experimental_test_set,
        "grid_size": 60,
        "validation_size": 805,
        "epochs": 25,
        "patch_indices": (0, 915, 1830, 2745, 3599),
        "stitcher": lambda: ExperimentalImageStitcher(
            point_size=3, patch_size=64, output_size=(60, 60)
        ),
        "normalize": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="ad_ltem",
        help="Dataset to use (default: ad_ltem, the noisy simulation).",
    )
    parser.add_argument(
        "--mode",
        choices=("test", "train", "generate"),
        default="test",
        help="Operation to run (default: test).",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help="Checkpoint(s); several checkpoints are averaged in test mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--representation",
        choices=tuple(REPRESENTATIONS),
        default="dc_dre",
        help="Input representation for training (default: dc_dre).",
    )
    parser.add_argument(
        "--stack-views",
        action="store_true",
        help="Give multi-view inputs to one encoder as channels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)

    generation = parser.add_argument_group("AD_LTEM data generation")
    generation.add_argument(
        "--amplitude",
        type=Path,
        default=SIMULATION_ASSETS / "amplitude.tiff",
    )
    generation.add_argument(
        "--phase", type=Path, default=SIMULATION_ASSETS / "phase.tiff"
    )
    generation.add_argument(
        "--probe", type=Path, default=SIMULATION_ASSETS / "probe.npy"
    )
    generation.add_argument("--image-size", type=int, default=512)
    generation.add_argument("--overlap-rate", type=float, default=75)
    generation.add_argument("--probe-block-size", type=int, default=4)
    generation.add_argument("--noise-level", type=float, default=0.01)
    return parser.parse_args()


def run_name(args):
    return args.representation + ("_stacked" if args.stack_views else "")


def run_train(args):
    settings = DATASETS[args.dataset]
    checkpoint = (
        args.checkpoint[0]
        if args.checkpoint
        else REPO_ROOT
        / "runs"
        / args.dataset
        / run_name(args)
        / f"seed_{args.seed}"
        / "model.pth"
    )
    prepared_data = settings["prepare"](
        args.data_root,
        grid_size=settings["grid_size"],
        validation_size=settings["validation_size"],
        seed=args.seed,
    )
    model = train_model(
        prepared_data,
        dataset_name=args.dataset,
        checkpoint_path=checkpoint,
        representation=args.representation,
        stack_views=args.stack_views,
        epochs=args.epochs or settings["epochs"],
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    amp_pred, phase_pred = reconstruct_patches(
        model,
        prepared_data.test_set.diffraction,
        prepared_data.amplitude_range,
        prepared_data.phase_range,
        args.batch_size,
    )
    result = evaluate_reconstruction(
        prepared_data.test_set,
        amp_pred,
        phase_pred,
        prepared_data.amplitude_range,
        prepared_data.phase_range,
        settings["stitcher"](),
        normalize=settings["normalize"],
    )
    print_results(args.dataset, [result])


def print_results(dataset_name, results):
    """Print patch and stitched metrics; several runs are averaged."""

    def mean_std(values, key):
        values = np.asarray(values)
        digits = {"SSIM": ".4f", "PSNR": ".2f"}.get(key, ".3e")
        return f"{values.mean():{digits}} ± {values.std(ddof=1):{digits}}"

    suffix = f" (mean ± std over {len(results)} runs)" if len(results) > 1 else ""
    if results[0]["patch"] is not None:
        if len(results) == 1:
            print(f"Per-patch metrics of the {dataset_name} test set (mean ± std over patches)")
            for name in ("amplitude", "phase"):
                print(f"{name.capitalize():10s} {format_average_metrics(results[0]['patch'][name])}")
        else:
            print(f"Per-patch metrics of the {dataset_name} test set{suffix}")
            for name in ("amplitude", "phase"):
                cells = [
                    f"{key} {mean_std([r['patch'][name][key]['mean'] for r in results], key)}"
                    for key in ("MSE", "MAE", "SSIM", "PSNR")
                ]
                print(f"{name.capitalize():10s} {' | '.join(cells)}")
    print(f"Stitched metrics of the {dataset_name} test region{suffix}")
    for name in ("amplitude", "phase"):
        if len(results) == 1:
            print(f"{name.capitalize():10s} {format_metrics(results[0]['stitched'][name])}")
        else:
            cells = [
                f"{key} {mean_std([r['stitched'][name][key] for r in results], key)}"
                for key in ("MSE", "MAE", "SSIM", "PSNR")
            ]
            print(f"{name.capitalize():10s} {' | '.join(cells)}")


def run_test(args):
    settings = DATASETS[args.dataset]
    output_dir = args.output_dir or REPO_ROOT / "results" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or select_device()
    checkpoints = args.checkpoint or [
        REPO_ROOT / "checkpoints" / args.dataset / "model.pth"
    ]

    test_data = settings["load_test"](
        args.data_root, grid_size=settings["grid_size"]
    )
    results = []
    for path in checkpoints:
        model, checkpoint = load_pretrained_model(path, device=device)
        amp_pred, phase_pred = reconstruct_patches(
            model,
            test_data.diffraction,
            checkpoint["amplitude_range"],
            checkpoint["phase_range"],
            args.batch_size,
            device,
        )
        results.append(
            evaluate_reconstruction(
                test_data,
                amp_pred,
                phase_pred,
                checkpoint["amplitude_range"],
                checkpoint["phase_range"],
                settings["stitcher"](),
                normalize=settings["normalize"],
            )
        )
        print(
            f"Checkpoint {path}: representation={checkpoint['representation']}"
            f"{' (stacked views)' if checkpoint.get('stack_views') else ''}"
        )
        if len(checkpoints) == 1:
            np.save(output_dir / "amplitude.npy", amp_pred)
            np.save(output_dir / "phase.npy", phase_pred)
    print_results(args.dataset, results)
    if len(checkpoints) > 1:
        return

    patch_figure, _ = plot_patch_comparison(
        test_data.diffraction,
        test_data.amplitude,
        amp_pred,
        test_data.phase,
        phase_pred,
        checkpoint["amplitude_range"],
        checkpoint["phase_range"],
        indices=settings["patch_indices"],
        grid_size=test_data.grid_size,
        save_path=output_dir / "patch_comparison.png",
    )
    images = results[0]["images"]
    global_figure = plot_global_stitching(
        images["amplitude_true"],
        images["amplitude_pred"],
        images["phase_true"],
        images["phase_pred"],
        results[0]["stitched"]["amplitude"],
        results[0]["stitched"]["phase"],
        save_path=output_dir / "global_stitching.png",
    )
    plt.close(patch_figure)
    plt.close(global_figure)
    print(f"Results saved to {output_dir}")


def run_generate(args):
    if args.dataset == "experimental":
        raise ValueError("Data generation is only available for AD_LTEM.")
    output_dir = args.output_dir or args.data_root
    generate_ad_ltem_dataset(
        amplitude_path=args.amplitude,
        phase_path=args.phase,
        probe_path=args.probe,
        output_dir=output_dir,
        image_size=(args.image_size, args.image_size),
        overlap_rate=args.overlap_rate,
        probe_block_size=(args.probe_block_size, args.probe_block_size),
        noise_level=args.noise_level,
        seed=args.seed,
    )


def main():
    args = parse_args()
    disable_tf32()
    operations = {
        "train": run_train,
        "test": run_test,
        "generate": run_generate,
    }
    operations[args.mode](args)


if __name__ == "__main__":
    main()
