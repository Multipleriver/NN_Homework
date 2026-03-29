import argparse
import importlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_FEATURE_COLS = [
    "cement",
    "slag",
    "flyash",
    "water",
    "superplasticizer",
    "coarseaggregate",
    "fineaggregate",
    "age",
]
TARGET_COL = "csMPa"


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_hidden_dims(hidden_dims_text: str) -> Tuple[int, ...]:
    dims = tuple(int(x.strip()) for x in hidden_dims_text.split(",") if x.strip())
    if not dims:
        raise ValueError("--hidden-dims must have at least one integer, e.g. 64,32")
    return dims


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def step1_initialize(project_dir: Path, data_path: Path, seed: int) -> None:
    set_seed(seed)
    print(f"[Step 1] Seed set to: {seed}")
    print(f"[Step 1] Project directory: {project_dir}")
    print(f"[Step 1] Data path: {data_path}")

    if data_path.exists() and data_path.is_file():
        print("[Step 1] Data path check: PASS")
    else:
        raise FileNotFoundError(f"[Step 1] Data path check: FAIL -> {data_path}")


def step2_data_check(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    print("[Step 2] Data shape:", df.shape)
    print("[Step 2] Columns:", list(df.columns))
    print("[Step 2] Dtypes:")
    print(df.dtypes)

    if df.shape[0] != 1030:
        raise ValueError(f"[Step 2] Sample count mismatch: expected 1030, got {df.shape[0]}")
    print("[Step 2] Sample count check: PASS (1030)")

    if list(df.columns) != BASE_FEATURE_COLS + [TARGET_COL]:
        raise ValueError("[Step 2] Column name/order check: FAIL")
    print("[Step 2] Column check: PASS")

    print("[Step 2] Missing value counts:")
    print(df.isna().sum())

    numeric_cols = df.columns.tolist()
    print("[Step 2] Negative value counts:")
    print((df[numeric_cols] < 0).sum())

    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (df[numeric_cols] < (q1 - 1.5 * iqr)) | (df[numeric_cols] > (q3 + 1.5 * iqr))
    print("[Step 2] IQR outlier counts:")
    print(outlier_mask.sum())

    return df


def add_engineered_features(df: pd.DataFrame, add_water_binder_ratio: bool, eps: float) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = list(BASE_FEATURE_COLS)
    if not add_water_binder_ratio:
        return df.copy(), feature_cols

    df_ext = df.copy()
    binder = df_ext["cement"] + df_ext["slag"] + df_ext["flyash"] + eps
    df_ext["water_binder_ratio"] = df_ext["water"] / binder
    feature_cols.append("water_binder_ratio")
    print("[Feature] Added water_binder_ratio = water/(cement+slag+flyash+eps)")
    print(
        "[Feature] water_binder_ratio stats -> "
        f"mean={df_ext['water_binder_ratio'].mean():.6f}, "
        f"std={df_ext['water_binder_ratio'].std():.6f}"
    )
    return df_ext, feature_cols


def step3_split_data(df: pd.DataFrame, feature_cols: Sequence[str], shuffle_before_split: bool, seed: int) -> Dict[str, pd.DataFrame | int]:
    n_samples = len(df)
    split_idx = int(np.floor(0.8 * n_samples))

    if shuffle_before_split:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print("[Step 3] Shuffle before split: ENABLED")
    else:
        print("[Step 3] Shuffle before split: DISABLED")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"[Step 3] Total samples: {n_samples}")
    print(f"[Step 3] Split index: {split_idx}")
    print(f"[Step 3] Train samples: {len(train_df)}")
    print(f"[Step 3] Test samples: {len(test_df)}")

    if len(train_df) + len(test_df) != n_samples:
        raise ValueError("[Step 3] Split count check: FAIL")
    if list(train_df[list(feature_cols)].columns) != list(test_df[list(feature_cols)].columns):
        raise ValueError("[Step 3] Feature order consistency check: FAIL")

    print("[Step 3] Feature order consistency check: PASS")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "split_idx": split_idx,
    }


def step4_preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[str, np.ndarray | SimpleImputer | StandardScaler]:
    x_train_raw = train_df[list(feature_cols)].values
    x_test_raw = test_df[list(feature_cols)].values

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    x_train_imputed = imputer.fit_transform(x_train_raw)
    x_test_imputed = imputer.transform(x_test_raw)

    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)

    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    if not np.isfinite(x_train_scaled).all() or not np.isfinite(x_test_scaled).all():
        raise ValueError("[Step 4] NaN/Inf check: FAIL")

    print("[Step 4] Imputer fit scope: TRAIN only")
    print("[Step 4] Scaler fit scope: TRAIN only")
    print("[Step 4] NaN/Inf check: PASS")
    print(f"[Step 4] X_train shape: {x_train_scaled.shape}, X_test shape: {x_test_scaled.shape}")

    return {
        "x_train": x_train_scaled.astype(np.float32),
        "x_test": x_test_scaled.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        "imputer": imputer,
        "scaler": scaler,
    }


def step5_build_train_val_test_tensors(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    val_ratio: float,
) -> Dict[str, torch.Tensor | DataLoader | StandardScaler | int]:
    n_train = x_train.shape[0]
    val_size = max(1, int(np.floor(n_train * val_ratio)))
    train_size = n_train - val_size
    if train_size < 2:
        raise ValueError("[Step 5] Train/val split invalid. Increase train size or reduce val_ratio.")

    x_subtrain = x_train[:train_size]
    y_subtrain = y_train[:train_size]
    x_val = x_train[train_size:]
    y_val = y_train[train_size:]

    y_scaler = StandardScaler()
    y_subtrain_scaled = y_scaler.fit_transform(y_subtrain.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)

    x_subtrain_tensor = torch.tensor(x_subtrain, dtype=torch.float32)
    y_subtrain_tensor = torch.tensor(y_subtrain_scaled, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(x_subtrain_tensor, y_subtrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sample_x, sample_y = next(iter(train_loader))
    print(f"[Step 5] Sub-train count: {train_size}, Val count: {val_size}, Test count: {len(x_test)}")
    print(f"[Step 5] X_subtrain dtype/shape: {x_subtrain_tensor.dtype}, {tuple(x_subtrain_tensor.shape)}")
    print(f"[Step 5] y_subtrain(dtype=scaled)/shape: {y_subtrain_tensor.dtype}, {tuple(y_subtrain_tensor.shape)}")
    print(f"[Step 5] DataLoader batch sample shape: X={tuple(sample_x.shape)}, y={tuple(sample_y.shape)}")

    return {
        "x_subtrain_tensor": x_subtrain_tensor,
        "y_subtrain_tensor": y_subtrain_tensor,
        "x_val_tensor": x_val_tensor,
        "y_val_tensor": y_val_tensor,
        "x_test_tensor": x_test_tensor,
        "y_test_tensor": y_test_tensor,
        "train_loader": train_loader,
        "y_scaler": y_scaler,
        "train_size": train_size,
        "val_size": val_size,
    }


def step6_build_model(
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    device: torch.device,
) -> MLPRegressor:
    model = MLPRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    dummy_x = torch.randn(4, input_dim, device=device)
    dummy_out = model(dummy_x)
    print(f"[Step 6] Model forward output shape check: {tuple(dummy_out.shape)}")
    return model


def step7_setup_training(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    loss_type: str,
    huber_delta: float,
    lr_patience: int,
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    if loss_type == "huber":
        criterion: nn.Module = nn.HuberLoss(delta=huber_delta)
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-6,
    )

    if loss_type == "huber":
        print(f"[Step 7] Loss function: HuberLoss(delta={huber_delta})")
    else:
        print("[Step 7] Loss function: MSELoss")
    print(f"[Step 7] Optimizer: Adam, lr={learning_rate}, weight_decay={weight_decay}")
    print(f"[Step 7] Scheduler: ReduceLROnPlateau, patience={lr_patience}, factor=0.5")

    return criterion, optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        n_samples += batch_x.size(0)

    return total_loss / max(n_samples, 1)


def evaluate_scaled_loss(
    model: nn.Module,
    x_tensor: torch.Tensor,
    y_scaled_tensor: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor.to(device))
        loss = criterion(preds, y_scaled_tensor.to(device))
    return float(loss.item())


def onnx_inference_self_check(
    model: nn.Module,
    onnx_path: Path,
    sample_input: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, float]:
    try:
        ort = importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise ImportError(
            "ONNX self-check requires onnxruntime. Install it with: pip install onnxruntime"
        ) from exc

    model.eval()
    with torch.no_grad():
        torch_out = model(sample_input).detach().cpu().numpy()

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.detach().cpu().numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    passed = bool(np.allclose(torch_out, ort_out, atol=atol, rtol=rtol))
    if not passed:
        raise RuntimeError(
            f"ONNX self-check failed: max_abs_diff={max_abs_diff:.6e}, atol={atol}, rtol={rtol}"
        )

    return {
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def step8_train_and_save(
    model: nn.Module,
    hidden_dims: Sequence[int],
    dropout: float,
    loader: DataLoader,
    x_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    train_loss_history: List[float],
    val_loss_history: List[float],
    early_stop_patience: int,
    min_delta: float,
) -> Dict[str, Path | List[float] | int | float]:
    if val_loss_history:
        best_val_loss = min(val_loss_history)
        best_epoch = int(np.argmin(val_loss_history) + 1)
    else:
        best_val_loss = float("inf")
        best_epoch = 0

    best_state = deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(len(train_loss_history) + 1, epochs + 1):
        epoch_train_loss = train_one_epoch(model, loader, criterion, optimizer, device)
        epoch_val_loss = evaluate_scaled_loss(model, x_val_tensor, y_val_tensor, criterion, device)

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[Step 8] Epoch {epoch:03d}/{epochs}, "
                f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, LR: {current_lr:.2e}"
            )

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"[Step 8] Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"[Step 8] Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}")

    model_path = output_dir / "final_model.pth"
    torch.save(model.state_dict(), model_path)

    onnx_path = output_dir / "final_model.onnx"
    model.eval()
    dummy_input = torch.randn(1, loader.dataset.tensors[0].shape[1], device=device)
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

    check_result = onnx_inference_self_check(model=model, onnx_path=onnx_path, sample_input=dummy_input)

    loss_path = output_dir / "train_loss_history.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": train_loss_history,
                "val_loss": val_loss_history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )

    reloaded_model = MLPRegressor(
        input_dim=loader.dataset.tensors[0].shape[1], hidden_dims=hidden_dims, dropout=dropout
    )
    reloaded_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    print(f"[Step 8] Model saved: {model_path}")
    print(f"[Step 8] ONNX saved: {onnx_path}")
    print(
        "[Step 8] ONNX self-check: PASS "
        f"(max_abs_diff={check_result['max_abs_diff']:.6e}, "
        f"atol={check_result['atol']:.1e}, rtol={check_result['rtol']:.1e})"
    )
    print(f"[Step 8] Loss history saved: {loss_path}")
    print("[Step 8] Model reload check: PASS")

    return {
        "model_path": model_path,
        "onnx_path": onnx_path,
        "loss_path": loss_path,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def step9_evaluate(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test_raw: torch.Tensor,
    y_scaler: StandardScaler,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, np.ndarray | float | Path | Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        preds_scaled = model(x_test.to(device)).cpu().numpy().reshape(-1, 1)

    preds = y_scaler.inverse_transform(preds_scaled).reshape(-1)
    y_true = y_test_raw.cpu().numpy().reshape(-1)

    mse = mean_squared_error(y_true, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Step 9] MSE: {metrics['mse']:.6f}")
    print(f"[Step 9] RMSE: {metrics['rmse']:.6f}")
    print(f"[Step 9] MAE: {metrics['mae']:.6f}")
    print(f"[Step 9] R2: {metrics['r2']:.6f}")
    print(f"[Step 9] Metrics saved: {metrics_path}")

    return {
        "y_true": y_true,
        "y_pred": preds,
        "metrics": metrics,
        "metrics_path": metrics_path,
    }


def step10_visualize(
    train_loss_history: List[float],
    val_loss_history: List[float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    loss_type: str,
) -> Dict[str, Path]:
    loss_fig = output_dir / "train_loss_curve.png"
    scatter_fig = output_dir / "test_true_vs_pred.png"

    loss_label = "MSE Loss (target scaled)" if loss_type == "mse" else "Huber Loss (target scaled)"

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_loss_history) + 1), train_loss_history, label="Sub-train Loss")
    plt.plot(np.arange(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel(loss_label)
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Predictions")
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal y=x")
    plt.xlabel("True Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title("True vs Predicted on Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_fig, dpi=150)
    plt.close()

    print(f"[Step 10] Saved loss curve: {loss_fig}")
    print(f"[Step 10] Saved scatter plot: {scatter_fig}")

    return {
        "loss_curve_path": loss_fig,
        "scatter_plot_path": scatter_fig,
    }


def step11_finalize_summary(
    output_dir: Path,
    seed: int,
    split_idx: int,
    train_count: int,
    test_count: int,
    subtrain_count: int,
    val_count: int,
    feature_cols: Sequence[str],
    shuffle_before_split: bool,
    add_water_binder_ratio: bool,
    hidden_dims: Sequence[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    val_ratio: float,
    early_stop_patience: int,
    lr_patience: int,
    loss_type: str,
    huber_delta: float,
    best_epoch: int,
    best_val_loss: float,
    metrics: Dict[str, float],
    model_path: Path,
    onnx_path: Path,
    loss_curve_path: Path,
    scatter_plot_path: Path,
) -> Path:
    summary_path = output_dir / "final_summary.txt"

    loss_desc = f"HuberLoss(delta={huber_delta})" if loss_type == "huber" else "MSELoss"

    lines = [
        "Homework1 Modified V2 Summary",
        "============================",
        f"Seed: {seed}",
        f"Split policy: 80/20 with shuffle_before_split={shuffle_before_split}",
        f"Split index: {split_idx}",
        f"Train count: {train_count}",
        f"Test count: {test_count}",
        f"Sub-train count: {subtrain_count}",
        f"Validation count: {val_count}",
        f"Validation split in train: last {int(val_ratio * 100)}% for val",
        f"Features: {list(feature_cols)}",
        f"Target: {TARGET_COL}",
        f"Feature engineering - water_binder_ratio: {add_water_binder_ratio}",
        f"Model hidden dims: {tuple(hidden_dims)}",
        f"Model dropout: {dropout}",
        f"Loss: {loss_desc}",
        f"Learning rate: {learning_rate}",
        f"Weight decay: {weight_decay}",
        f"Batch size: {batch_size}",
        f"Max epochs: {epochs}",
        f"Scheduler: ReduceLROnPlateau(patience={lr_patience}, factor=0.5)",
        f"Early stopping patience: {early_stop_patience}",
        f"Best epoch: {best_epoch}",
        f"Best validation loss: {best_val_loss:.6f}",
        "Target scaling: StandardScaler fitted on sub-train target only",
        f"Test MSE: {metrics['mse']:.6f}",
        f"Test RMSE: {metrics['rmse']:.6f}",
        f"Test MAE: {metrics['mae']:.6f}",
        f"Test R2: {metrics['r2']:.6f}",
        f"Model path: {model_path}",
        f"ONNX path: {onnx_path}",
        f"Loss curve path: {loss_curve_path}",
        f"Scatter plot path: {scatter_plot_path}",
        "Reproducibility notes:",
        "- Fixed random seed for random/NumPy/PyTorch.",
        "- Feature preprocessing fit only on train set.",
        "- Target scaler fit only on sub-train target.",
        "- Script: train_modified_v2.py",
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Step 11] Final summary saved: {summary_path}")
    return summary_path


def run_pipeline(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path)

    output_dir_arg = Path(args.output_dir)
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else project_dir / output_dir_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = parse_hidden_dims(args.hidden_dims)

    if not (1 <= args.step <= 11):
        raise ValueError("--step must be in [1, 11]")

    step1_initialize(project_dir=project_dir, data_path=data_path, seed=args.seed)
    if args.step == 1:
        return

    df = step2_data_check(data_path=data_path)
    if args.step == 2:
        return

    df, feature_cols = add_engineered_features(
        df=df,
        add_water_binder_ratio=args.add_water_binder_ratio,
        eps=args.wb_eps,
    )

    split_artifacts = step3_split_data(
        df=df,
        feature_cols=feature_cols,
        shuffle_before_split=args.shuffle_before_split,
        seed=args.seed,
    )
    train_df = split_artifacts["train_df"]
    test_df = split_artifacts["test_df"]
    if args.step == 3:
        return

    preprocess_artifacts = step4_preprocess(train_df=train_df, test_df=test_df, feature_cols=feature_cols)
    if args.step == 4:
        return

    tensor_artifacts = step5_build_train_val_test_tensors(
        x_train=preprocess_artifacts["x_train"],
        y_train=preprocess_artifacts["y_train"],
        x_test=preprocess_artifacts["x_test"],
        y_test=preprocess_artifacts["y_test"],
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )
    if args.step == 5:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device selected: {device}")

    model = step6_build_model(
        input_dim=tensor_artifacts["x_subtrain_tensor"].shape[1],
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        device=device,
    )
    if args.step == 6:
        return

    criterion, optimizer, scheduler = step7_setup_training(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        lr_patience=args.lr_patience,
    )

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    one_epoch_train_loss = train_one_epoch(
        model=model,
        loader=tensor_artifacts["train_loader"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    one_epoch_val_loss = evaluate_scaled_loss(
        model=model,
        x_tensor=tensor_artifacts["x_val_tensor"],
        y_scaled_tensor=tensor_artifacts["y_val_tensor"],
        criterion=criterion,
        device=device,
    )
    scheduler.step(one_epoch_val_loss)

    train_loss_history.append(one_epoch_train_loss)
    val_loss_history.append(one_epoch_val_loss)
    print(
        f"[Step 7] Single-epoch losses -> "
        f"train: {one_epoch_train_loss:.6f}, val: {one_epoch_val_loss:.6f}, "
        f"lr: {optimizer.param_groups[0]['lr']:.2e}"
    )
    if args.step == 7:
        return

    train_artifacts = step8_train_and_save(
        model=model,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        loader=tensor_artifacts["train_loader"],
        x_val_tensor=tensor_artifacts["x_val_tensor"],
        y_val_tensor=tensor_artifacts["y_val_tensor"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=output_dir,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
    )
    if args.step == 8:
        return

    eval_artifacts = step9_evaluate(
        model=model,
        x_test=tensor_artifacts["x_test_tensor"],
        y_test_raw=tensor_artifacts["y_test_tensor"],
        y_scaler=tensor_artifacts["y_scaler"],
        device=device,
        output_dir=output_dir,
    )
    if args.step == 9:
        return

    figure_artifacts = step10_visualize(
        train_loss_history=train_artifacts["train_loss_history"],
        val_loss_history=train_artifacts["val_loss_history"],
        y_true=eval_artifacts["y_true"],
        y_pred=eval_artifacts["y_pred"],
        output_dir=output_dir,
        loss_type=args.loss_type,
    )
    if args.step == 10:
        return

    step11_finalize_summary(
        output_dir=output_dir,
        seed=args.seed,
        split_idx=split_artifacts["split_idx"],
        train_count=len(train_df),
        test_count=len(test_df),
        subtrain_count=tensor_artifacts["train_size"],
        val_count=tensor_artifacts["val_size"],
        feature_cols=feature_cols,
        shuffle_before_split=args.shuffle_before_split,
        add_water_binder_ratio=args.add_water_binder_ratio,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
        lr_patience=args.lr_patience,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        best_epoch=train_artifacts["best_epoch"],
        best_val_loss=train_artifacts["best_val_loss"],
        metrics=eval_artifacts["metrics"],
        model_path=train_artifacts["model_path"],
        onnx_path=train_artifacts["onnx_path"],
        loss_curve_path=figure_artifacts["loss_curve_path"],
        scatter_plot_path=figure_artifacts["scatter_plot_path"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework1 modified v2 regression pipeline")
    parser.add_argument("--step", type=int, default=11, help="Current step to execute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--shuffle-before-split", action="store_true", help="Shuffle full dataset before 80/20 split")

    parser.add_argument("--loss-type", type=str, default="huber", choices=["huber", "mse"], help="Training loss")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber loss delta")

    parser.add_argument("--hidden-dims", type=str, default="64,32", help="Comma-separated hidden dims, e.g. 256,128,64")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for hidden layers")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")

    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio inside train split")
    parser.add_argument("--early-stop-patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--lr-patience", type=int, default=10, help="LR scheduler patience")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum val loss improvement")

    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training max epochs")

    parser.add_argument("--add-water-binder-ratio", action="store_true", help="Add water/binder engineered feature")
    parser.add_argument("--wb-eps", type=float, default=1e-6, help="Epsilon for water/binder denominator")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_modified_v2",
        help="Output directory path (absolute or relative to project dir)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "resource" / "Concrete_Data_Yeh.csv"),
        help="Path to Concrete_Data_Yeh.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
