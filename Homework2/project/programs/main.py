from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from dataset import build_dataloaders, validate_batch_shape_and_labels
from engine import benchmark_amp_speed, train_model
from evaluate import render_all_plots
from model import count_trainable_parameters, create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_model_names(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--models cannot be empty")

    valid = {"baseline", "resnet18", "resnet"}
    for model_name in models:
        if model_name not in valid:
            raise ValueError(f"Unsupported model in --models: {model_name}")

    normalized = ["resnet18" if name == "resnet" else name for name in models]
    deduped = []
    for name in normalized:
        if name not in deduped:
            deduped.append(name)
    return deduped


def get_device_info() -> tuple[torch.device, dict]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        return device, {
            "device": "cuda",
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda,
        }

    return torch.device("cpu"), {
        "device": "cpu",
        "name": "CPU",
        "total_memory_gb": None,
        "capability": None,
        "cuda_version": torch.version.cuda,
    }


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def _run_training_suite(
    model_names: list[str],
    train_loader,
    test_loader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp_enabled: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_result: dict = {}
    for model_name in model_names:
        model = create_model(model_name=model_name, num_classes=10)
        param_count = count_trainable_parameters(model)

        result = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            amp_enabled=amp_enabled,
        )

        suite_result[model_name] = {
            "params": int(param_count),
            "metrics_path": str(result.metrics_path),
            "history_path": str(result.history_path),
            "final_model_path": str(result.final_model_path),
            "best_model_path": str(result.best_model_path),
            "best_epoch": int(result.best_epoch),
            "best_test_acc": float(result.best_test_acc),
            "best_test_loss": float(result.best_test_loss),
            "final_test_acc": float(result.final_test_acc),
            "final_test_loss": float(result.final_test_loss),
            "max_gpu_memory_mb": float(result.max_gpu_memory_mb),
        }

    return suite_result


def _check_dry_run_convergence(dryrun_dir: Path, model_names: list[str]) -> dict:
    verdict = {}
    for model_name in model_names:
        history = _load_json(dryrun_dir / f"{model_name}_history.json")
        start_loss = float(history["train_loss"][0])
        end_loss = float(history["train_loss"][-1])
        verdict[model_name] = {
            "start_train_loss": start_loss,
            "end_train_loss": end_loss,
            "loss_decreased": bool(end_loss < start_loss),
        }
    return verdict


def _export_model_to_onnx(
    model_name: str,
    checkpoint_path: Path,
    onnx_path: Path,
    num_classes: int = 10,
) -> Path:
    # Export from best checkpoint on CPU for reproducible ONNX artifacts.
    model = create_model(model_name=model_name, num_classes=num_classes).to("cpu")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    return onnx_path


def _export_models_to_onnx(model_names: list[str], full_run_summary: dict, report_dir: Path) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    for model_name in model_names:
        checkpoint_path = Path(full_run_summary[model_name]["best_model_path"])
        onnx_path = report_dir / f"{model_name}.onnx"
        exported_paths.append(
            _export_model_to_onnx(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                onnx_path=onnx_path,
            )
        )
    return exported_paths


def _write_final_summary(
    output_dir: Path,
    report_dir: Path,
    run_config: dict,
    device_info: dict,
    step1_check: dict,
    dry_run_verdict: dict,
    full_run_summary: dict,
    best_model_name: str,
    amp_benchmark: dict,
    generated_figures: list[Path],
) -> Path:
    summary_path = output_dir / "final_summary.txt"

    lines = [
        "Homework2 Summary",
        "=================",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Seed: {run_config['seed']}",
        f"Models: {run_config['models']}",
        f"Epochs(full): {run_config['epochs']}",
        f"Epochs(dry-run): {run_config['dry_run_epochs']}",
        f"Batch size: {run_config['batch_size']}",
        f"Learning rate: {run_config['learning_rate']}",
        f"Weight decay: {run_config['weight_decay']}",
        f"Device: {device_info['device']} ({device_info['name']})",
        f"AMP enabled: {run_config['amp_enabled']}",
        "",
        "Step 1 validation:",
        f"- batch_shape: {step1_check['batch_shape']}",
        f"- label_min/max: {step1_check['label_min']} / {step1_check['label_max']}",
        "",
        "Step 2 dry-run verdict:",
    ]

    for model_name, item in dry_run_verdict.items():
        lines.append(
            f"- {model_name}: loss {item['start_train_loss']:.4f} -> {item['end_train_loss']:.4f}, "
            f"decreased={item['loss_decreased']}"
        )

    lines.extend(["", "Step 3 full training:"])
    for model_name, item in full_run_summary.items():
        lines.append(
            f"- {model_name}: best_acc={item['best_test_acc']:.2f}%, best_loss={item['best_test_loss']:.4f}, "
            f"best_epoch={item['best_epoch']}, params={item['params']}"
        )

    lines.extend(
        [
            "",
            f"Best model by test accuracy: {best_model_name}",
            f"Primary checkpoint: {output_dir / 'final_model.pth'}",
            "",
            "AMP benchmark:",
        ]
    )

    if amp_benchmark.get("available"):
        lines.extend(
            [
                f"- FP32 throughput: {amp_benchmark['fp32']['samples_per_sec']:.2f} samples/s",
                f"- AMP throughput: {amp_benchmark['amp']['samples_per_sec']:.2f} samples/s",
                f"- Speedup: {amp_benchmark['speedup_x']:.3f}x",
                f"- AMP memory / FP32 memory: {amp_benchmark['amp_memory_vs_fp32']:.3f}",
            ]
        )
    else:
        lines.append(f"- skipped: {amp_benchmark.get('reason', 'unknown reason')}")

    lines.extend(["", "Generated figures:"])
    for fig in generated_figures:
        lines.append(f"- {fig}")

    lines.append(f"\nReport path: {report_dir / 'homework2_report.md'}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _resolve_report_path(report_dir: Path) -> Path:
    return report_dir / "homework2_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework2 full automation pipeline")
    parser.add_argument("--step", type=int, default=4, help="1=data check, 2=dry run, 3=train+plot, 4=full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="baseline,resnet18")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dry-run-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--num-workers", type=int, default=-1, help="-1 means auto infer")
    parser.add_argument("--train-limit", type=int, default=0, help="Use >0 for debugging")
    parser.add_argument("--test-limit", type=int, default=0, help="Use >0 for debugging")

    parser.add_argument("--download-if-missing", action="store_true")

    parser.set_defaults(amp=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    parser.set_defaults(run_amp_benchmark=True)
    parser.add_argument("--no-amp-benchmark", dest="run_amp_benchmark", action="store_false")
    parser.add_argument("--amp-benchmark-steps", type=int, default=30)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step < 1 or args.step > 4:
        raise ValueError("--step must be in [1, 4]")

    effective_dry_run_epochs = max(int(args.dry_run_epochs), 3)
    effective_full_epochs = max(int(args.epochs), 3)
    if int(args.dry_run_epochs) < 3:
        print(f"[Config] dry-run epochs adjusted from {args.dry_run_epochs} to {effective_dry_run_epochs}.")
    if int(args.epochs) < 3:
        print(f"[Config] full-training epochs adjusted from {args.epochs} to {effective_full_epochs}.")

    set_seed(args.seed)

    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent
    resource_dir = project_dir / "resource"
    output_dir = project_dir / "output"
    report_dir = project_dir / "report"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_names = parse_model_names(args.models)

    device, device_info = get_device_info()
    amp_enabled = bool(args.amp and device.type == "cuda")

    data_bundle = build_dataloaders(
        resource_dir=resource_dir,
        batch_size=args.batch_size,
        requested_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        download_if_missing=args.download_if_missing,
        train_limit=(args.train_limit if args.train_limit > 0 else None),
        test_limit=(args.test_limit if args.test_limit > 0 else None),
    )

    step1_check = validate_batch_shape_and_labels(data_bundle.train_loader)
    step1_path = output_dir / "step1_data_validation.json"
    _dump_json(
        step1_path,
        {
            "step": 1,
            "resource_dir": str(resource_dir),
            "train_path": str(data_bundle.train_path),
            "test_path": str(data_bundle.test_path),
            "train_size": data_bundle.train_size,
            "test_size": data_bundle.test_size,
            "num_workers": data_bundle.num_workers,
            "batch_check": step1_check,
        },
    )
    print(f"[Step 1] PASS. Validation saved: {step1_path}")

    if args.step == 1:
        return

    dryrun_dir = output_dir / "dryrun"
    dryrun_summary = _run_training_suite(
        model_names=model_names,
        train_loader=data_bundle.train_loader,
        test_loader=data_bundle.test_loader,
        device=device,
        output_dir=dryrun_dir,
        epochs=effective_dry_run_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp_enabled=amp_enabled,
    )
    dry_run_verdict = _check_dry_run_convergence(dryrun_dir=dryrun_dir, model_names=model_names)
    _dump_json(output_dir / "step2_dryrun_summary.json", {"summary": dryrun_summary, "verdict": dry_run_verdict})

    print("[Step 2] Dry-run completed.")
    for model_name, verdict in dry_run_verdict.items():
        print(
            f"  - {model_name}: loss {verdict['start_train_loss']:.4f} -> {verdict['end_train_loss']:.4f}, "
            f"decreased={verdict['loss_decreased']}"
        )

    if args.step == 2:
        return

    amp_benchmark = {
        "available": False,
        "reason": "benchmark disabled",
    }
    if args.run_amp_benchmark:
        primary_model = model_names[-1]
        amp_benchmark = benchmark_amp_speed(
            model_builder=lambda: create_model(model_name=primary_model, num_classes=10),
            train_loader=data_bundle.train_loader,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps=args.amp_benchmark_steps,
        )
    _dump_json(output_dir / "amp_benchmark.json", amp_benchmark)

    full_run_summary = _run_training_suite(
        model_names=model_names,
        train_loader=data_bundle.train_loader,
        test_loader=data_bundle.test_loader,
        device=device,
        output_dir=output_dir,
        epochs=effective_full_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp_enabled=amp_enabled,
    )

    for model_name in model_names:
        metrics_path = output_dir / f"{model_name}_metrics.json"
        metrics = _load_json(metrics_path)
        metrics["params"] = full_run_summary[model_name]["params"]
        _dump_json(metrics_path, metrics)

    best_model_name = max(model_names, key=lambda name: full_run_summary[name]["best_test_acc"])
    best_ckpt = Path(full_run_summary[best_model_name]["best_model_path"])
    shutil.copyfile(best_ckpt, output_dir / "final_model.pth")

    print(f"[Step 3] Full training completed. Best model: {best_model_name}")

    generated_figures = render_all_plots(output_dir=output_dir, report_dir=report_dir)
    print(f"[Step 3] Generated {len(generated_figures)} figures in report directory.")

    exported_onnx_paths = _export_models_to_onnx(
        model_names=model_names,
        full_run_summary=full_run_summary,
        report_dir=report_dir,
    )
    for onnx_path in exported_onnx_paths:
        print(f"[Step 3] ONNX exported: {onnx_path}")

    if args.step == 3:
        return

    report_path = _resolve_report_path(report_dir=report_dir)

    summary_path = _write_final_summary(
        output_dir=output_dir,
        report_dir=report_dir,
        run_config={
            "seed": args.seed,
            "models": model_names,
            "epochs": effective_full_epochs,
            "dry_run_epochs": effective_dry_run_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "amp_enabled": amp_enabled,
        },
        device_info=device_info,
        step1_check=step1_check,
        dry_run_verdict=dry_run_verdict,
        full_run_summary=full_run_summary,
        best_model_name=best_model_name,
        amp_benchmark=amp_benchmark,
        generated_figures=generated_figures,
    )

    _dump_json(
        output_dir / "run_manifest.json",
        {
            "project_dir": str(project_dir),
            "resource_dir": str(resource_dir),
            "output_dir": str(output_dir),
            "report_dir": str(report_dir),
            "step1_validation": str(step1_path),
            "dryrun_summary": str(output_dir / "step2_dryrun_summary.json"),
            "amp_benchmark": str(output_dir / "amp_benchmark.json"),
            "report_path": str(report_path),
            "summary_path": str(summary_path),
            "onnx_exports": [str(path) for path in exported_onnx_paths],
            "best_model": best_model_name,
            "final_checkpoint": str(output_dir / "final_model.pth"),
        },
    )

    print(f"[Step 4] Report path recorded: {report_path}")
    print(f"[Step 4] Final summary generated: {summary_path}")


if __name__ == "__main__":
    main()
