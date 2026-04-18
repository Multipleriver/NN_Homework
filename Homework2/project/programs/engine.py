from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    model_name: str
    final_model_path: Path
    best_model_path: Path
    history_path: Path
    metrics_path: Path
    best_epoch: int
    best_test_acc: float
    best_test_loss: float
    final_train_acc: float
    final_test_acc: float
    final_train_loss: float
    final_test_loss: float
    max_gpu_memory_mb: float


def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    is_train: bool,
) -> dict:
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return {
        "loss": avg_loss,
        "acc": avg_acc,
        "samples": total_samples,
    }


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    amp_enabled: bool,
) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=learning_rate * 0.05,
    )
    scaler = GradScaler(enabled=amp_enabled)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    best_epoch = 0
    best_test_acc = -1.0
    best_test_loss = float("inf")
    best_state_dict = deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        train_stats = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            is_train=True,
        )

        with torch.no_grad():
            test_stats = _run_one_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                is_train=False,
            )

        scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_stats["loss"]))
        history["train_acc"].append(float(train_stats["acc"]))
        history["test_loss"].append(float(test_stats["loss"]))
        history["test_acc"].append(float(test_stats["acc"]))
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(float(epoch_seconds))

        if (test_stats["acc"] > best_test_acc) or (
            abs(test_stats["acc"] - best_test_acc) < 1e-12 and test_stats["loss"] < best_test_loss
        ):
            best_test_acc = float(test_stats["acc"])
            best_test_loss = float(test_stats["loss"])
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

        print(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.2f}% | "
            f"test_loss={test_stats['loss']:.4f} test_acc={test_stats['acc']:.2f}% | "
            f"lr={current_lr:.3e} | t={epoch_seconds:.2f}s"
        )

    final_model_path = output_dir / f"{model_name}_final_model.pth"
    best_model_path = output_dir / f"{model_name}_best_model.pth"
    history_path = output_dir / f"{model_name}_history.json"
    metrics_path = output_dir / f"{model_name}_metrics.json"

    torch.save(model.state_dict(), final_model_path)
    torch.save(best_state_dict, best_model_path)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    final_train_loss = float(history["train_loss"][-1])
    final_test_loss = float(history["test_loss"][-1])
    final_train_acc = float(history["train_acc"][-1])
    final_test_acc = float(history["test_acc"][-1])

    max_gpu_memory_mb = 0.0
    if device.type == "cuda":
        max_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024**2))

    metrics = {
        "model_name": model_name,
        "epochs": int(epochs),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "CrossEntropyLoss",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "amp_enabled": bool(amp_enabled),
        "best_epoch": int(best_epoch),
        "best_test_acc": float(best_test_acc),
        "best_test_loss": float(best_test_loss),
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "max_gpu_memory_mb": max_gpu_memory_mb,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainResult(
        model_name=model_name,
        final_model_path=final_model_path,
        best_model_path=best_model_path,
        history_path=history_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        best_test_acc=best_test_acc,
        best_test_loss=best_test_loss,
        final_train_acc=final_train_acc,
        final_test_acc=final_test_acc,
        final_train_loss=final_train_loss,
        final_test_loss=final_test_loss,
        max_gpu_memory_mb=max_gpu_memory_mb,
    )


def _benchmark_single_mode(
    model_builder: Callable[[], nn.Module],
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    amp_enabled: bool,
    max_steps: int,
) -> dict:
    model = model_builder().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize(device=device)

    consumed_samples = 0
    iterator = iter(train_loader)

    start = time.perf_counter()
    for _ in range(max_steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        consumed_samples += int(images.size(0))

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    elapsed = time.perf_counter() - start
    throughput = consumed_samples / max(elapsed, 1e-6)

    max_gpu_memory_mb = 0.0
    if device.type == "cuda":
        max_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device=device) / (1024**2))

    return {
        "amp_enabled": bool(amp_enabled),
        "steps": int(max_steps),
        "samples": int(consumed_samples),
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(throughput),
        "max_gpu_memory_mb": max_gpu_memory_mb,
    }


def benchmark_amp_speed(
    model_builder: Callable[[], nn.Module],
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_steps: int = 30,
) -> dict:
    if device.type != "cuda":
        return {
            "available": False,
            "reason": "CUDA device is not available, AMP benchmark skipped.",
        }

    fp32_stats = _benchmark_single_mode(
        model_builder=model_builder,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        amp_enabled=False,
        max_steps=max_steps,
    )

    amp_stats = _benchmark_single_mode(
        model_builder=model_builder,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        amp_enabled=True,
        max_steps=max_steps,
    )

    speedup = amp_stats["samples_per_sec"] / max(fp32_stats["samples_per_sec"], 1e-6)
    memory_ratio = amp_stats["max_gpu_memory_mb"] / max(fp32_stats["max_gpu_memory_mb"], 1e-6)

    return {
        "available": True,
        "fp32": fp32_stats,
        "amp": amp_stats,
        "speedup_x": float(speedup),
        "amp_memory_vs_fp32": float(memory_ratio),
    }
