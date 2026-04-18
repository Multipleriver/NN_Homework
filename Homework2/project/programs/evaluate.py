from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_history_files(output_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file in output_dir.glob("*_history.json"):
        model_name = file.name.replace("_history.json", "")
        mapping[model_name] = file
    return mapping


def discover_metrics_files(output_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file in output_dir.glob("*_metrics.json"):
        model_name = file.name.replace("_metrics.json", "")
        mapping[model_name] = file
    return mapping


def plot_learning_curves(history: dict, model_name: str, report_dir: Path) -> list[Path]:
    epochs = [int(v) for v in history["epoch"]]
    train_acc = history["train_acc"]
    test_acc = history["test_acc"]
    train_loss = history["train_loss"]
    test_loss = history["test_loss"]

    if not epochs:
        raise ValueError(f"{model_name} history has empty epoch array")

    expected_length = len(epochs)
    for key, values in [
        ("train_acc", train_acc),
        ("test_acc", test_acc),
        ("train_loss", train_loss),
        ("test_loss", test_loss),
    ]:
        if len(values) != expected_length:
            raise ValueError(
                f"{model_name} history length mismatch: epoch={expected_length}, {key}={len(values)}"
            )

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    marker = "o"

    axes[0].plot(
        epochs,
        train_acc,
        label="Train Accuracy",
        linewidth=2.2,
        color="#1f77b4",
        marker=marker,
        markersize=6,
    )
    axes[0].plot(
        epochs,
        test_acc,
        label="Test Accuracy",
        linewidth=2.2,
        color="#d62728",
        marker=marker,
        markersize=6,
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title(f"{model_name} Accuracy Curves")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_loss,
        label="Train Loss",
        linewidth=2.2,
        color="#2ca02c",
        marker=marker,
        markersize=6,
    )
    axes[1].plot(
        epochs,
        test_loss,
        label="Test Loss",
        linewidth=2.2,
        color="#ff7f0e",
        marker=marker,
        markersize=6,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CrossEntropy Loss")
    axes[1].set_title(f"{model_name} Loss Curves")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    axes[1].set_xticks(epochs)
    if len(epochs) == 1:
        center = float(epochs[0])
        axes[1].set_xlim(center - 0.5, center + 0.5)
    else:
        axes[1].set_xlim(min(epochs), max(epochs))

    fig.tight_layout()

    png_path = report_dir / f"{model_name}_curves.png"
    svg_path = report_dir / f"{model_name}_curves.svg"
    fig.savefig(png_path, dpi=240)
    fig.savefig(svg_path)
    plt.close(fig)

    return [png_path, svg_path]


def plot_model_comparison(metrics_map: dict[str, dict], report_dir: Path) -> list[Path]:
    model_names = list(metrics_map.keys())
    best_test_acc = [metrics_map[m]["best_test_acc"] for m in model_names]
    best_test_loss = [metrics_map[m]["best_test_loss"] for m in model_names]

    x = list(range(len(model_names)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.suptitle("Model Comparison (Separated Metrics)", fontsize=12)

    bars_acc = axes[0].bar(x, best_test_acc, width=0.62, color="#4c78a8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel("Best Test Accuracy (%)")
    axes[0].set_title("Accuracy Comparison")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    bars_loss = axes[1].bar(x, best_test_loss, width=0.62, color="#f58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel("Best Test Loss")
    axes[1].set_title("Loss Comparison")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    for bar in bars_acc:
        height = float(bar.get_height())
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars_loss:
        height = float(bar.get_height())
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    comparison_png_path = report_dir / "model_comparison_compact.png"
    comparison_svg_path = report_dir / "model_comparison_compact.svg"
    fig.savefig(comparison_png_path, dpi=240)
    fig.savefig(comparison_svg_path)
    plt.close(fig)

    obsolete_paths = [
        report_dir / "model_comparison_accuracy.png",
        report_dir / "model_comparison_loss.png",
    ]
    for obsolete_path in obsolete_paths:
        if obsolete_path.exists():
            obsolete_path.unlink()

    return [comparison_png_path, comparison_svg_path]


def validate_plot_files(paths: Iterable[Path], min_bytes: int = 1024) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Plot file missing: {path}")
        if path.stat().st_size < min_bytes:
            raise RuntimeError(f"Plot file looks empty or too small: {path}")


def render_all_plots(output_dir: Path, report_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    history_files = discover_history_files(output_dir)
    metrics_files = discover_metrics_files(output_dir)

    if not history_files:
        raise FileNotFoundError(f"No *_history.json files found in: {output_dir}")

    generated: list[Path] = []
    metrics_map: dict[str, dict] = {}

    for model_name, history_file in sorted(history_files.items()):
        history = _load_json(history_file)
        generated.extend(plot_learning_curves(history=history, model_name=model_name, report_dir=report_dir))

        metrics_file = metrics_files.get(model_name)
        if metrics_file is not None:
            metrics_map[model_name] = _load_json(metrics_file)

    if metrics_map:
        generated.extend(plot_model_comparison(metrics_map=metrics_map, report_dir=report_dir))

    validate_plot_files(generated)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render learning curves for Homework2 experiments")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory containing metrics JSON files")
    parser.add_argument("--report-dir", type=str, required=True, help="Directory to save figure files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = render_all_plots(output_dir=Path(args.output_dir), report_dir=Path(args.report_dir))
    print("Generated figures:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
