from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    learning_rate: float = 1e-3
    binn_weight_decay: float = 1e-2
    gradient_clip_norm: float = 5.0
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    # False: Adam with the L2 term added to the gradient (frozen recipe); True: AdamW.
    decoupled_weight_decay: bool = False


@dataclass(frozen=True)
class FitResult:
    model_id: str
    best_epoch: int
    best_validation_nll: float
    checkpoint_path: str
    history_path: str


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def initialize_baseline(model, targets: np.ndarray, mask: np.ndarray, train_index: np.ndarray) -> None:
    events = targets[train_index].sum(axis=0)
    at_risk = mask[train_index].sum(axis=0)
    hazard = np.clip((events + 0.5) / (at_risk + 1.0), 1e-5, 1.0 - 1e-5)
    baseline = np.log(hazard / (1.0 - hazard)).astype(np.float32)
    with torch.no_grad():
        model.baseline_logits.copy_(torch.from_numpy(baseline).to(model.baseline_logits.device))


def _optimizer(model, settings: TrainingSettings) -> Adam | AdamW:
    fusion = [
        *model.h6_bn.parameters(),
        *model.prs_bn.parameters(),
        *model.fusion.parameters(),
        model.baseline_logits,
    ]
    optimizer_class = AdamW if settings.decoupled_weight_decay else Adam
    return optimizer_class([
        {
            "params": model.binn.parameters(),
            "lr": settings.learning_rate,
            "weight_decay": settings.binn_weight_decay,
        },
        {"params": fusion, "lr": settings.learning_rate, "weight_decay": 0.0},
    ])


def validation_nll(
    model,
    omics: np.ndarray,
    prs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    index: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    loader = DataLoader(TensorDataset(
        torch.from_numpy(omics[index]),
        torch.from_numpy(prs[index]),
        torch.from_numpy(targets[index]),
        torch.from_numpy(mask[index]),
    ), batch_size=256, shuffle=False)
    with torch.no_grad():
        for x_batch, prs_batch, y_batch, mask_batch in loader:
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            logits = model(x_batch.to(device), prs_batch.to(device))["hazard_logits"]
            raw = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="none") * mask_batch
            total_loss += float(raw.sum())
            total_weight += float(mask_batch.sum())
    return total_loss / max(total_weight, 1.0)


def fit_model(
    model,
    omics: np.ndarray,
    prs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    device: torch.device,
    output_dir: str | Path,
    model_id: str,
    training_seed: int,
    settings: TrainingSettings,
    teacher_logits: np.ndarray | None = None,
) -> FitResult:
    """Fit Teacher, distilled Student, or supervised-only Student.

    If teacher_logits is None, the only loss is masked hard-label NLL. Otherwise
    total loss is (1-alpha)*hard + alpha*T^2*soft, with validation/early stopping
    always based on hard-label validation NLL.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initialize_baseline(model, targets, mask, train_index)
    optimizer = _optimizer(model, settings)
    tensors = [
        torch.from_numpy(omics[train_index]),
        torch.from_numpy(prs[train_index]),
        torch.from_numpy(targets[train_index]),
        torch.from_numpy(mask[train_index]),
    ]
    if teacher_logits is not None:
        tensors.append(torch.from_numpy(teacher_logits[train_index]))
    dataset = TensorDataset(*tensors)
    best_loss = float("inf")
    best_epoch = 0
    waiting = 0
    best_state = None
    history: list[dict[str, float | int]] = []
    offset = 1000 if teacher_logits is not None else 0
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        hard_values: list[float] = []
        soft_values: list[float] = []
        total_values: list[float] = []
        generator = torch.Generator().manual_seed(int(training_seed) + offset + epoch)
        loader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            generator=generator,
        )
        for batch in loader:
            x_batch, prs_batch, y_batch, mask_batch = [item.to(device) for item in batch[:4]]
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(x_batch, prs_batch)["hazard_logits"]
            hard = masked_bce(student_logits, y_batch, mask_batch)
            if teacher_logits is None:
                soft = torch.zeros((), device=device)
                total = hard
            else:
                teacher_batch = batch[4].to(device)
                temperature = settings.distillation_temperature
                teacher_soft = torch.sigmoid(teacher_batch / temperature)
                soft_raw = F.binary_cross_entropy_with_logits(
                    student_logits / temperature, teacher_soft, reduction="none"
                )
                soft = (soft_raw * mask_batch).sum() / mask_batch.sum().clamp_min(1.0)
                alpha = settings.distillation_alpha
                total = (1.0 - alpha) * hard + alpha * temperature**2 * soft
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            hard_values.append(float(hard.detach()))
            soft_values.append(float(soft.detach()))
            total_values.append(float(total.detach()))
        val_loss = validation_nll(
            model, omics, prs, targets, mask, validation_index, device
        )
        history.append({
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_values)),
            "train_hard_nll": float(np.mean(hard_values)),
            "train_soft_bce": float(np.mean(soft_values)),
            "validation_hard_nll": val_loss,
        })
        print(
            f"{model_id} seed={training_seed} epoch={epoch:03d} "
            f"train={np.mean(total_values):.6f} val_nll={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss - settings.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            waiting = 0
            best_state = copy.deepcopy({
                name: value.detach().cpu() for name, value in model.state_dict().items()
            })
        else:
            waiting += 1
            if waiting >= settings.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{model_id} did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    checkpoint = output / f"{model_id}_best_seed{training_seed}.pt"
    history_path = output / f"{model_id}_training_history_seed{training_seed}.csv"
    torch.save(best_state, checkpoint)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return FitResult(model_id, best_epoch, best_loss, str(checkpoint), str(history_path))


def settings_from_config(config, quick: bool = False) -> TrainingSettings:
    raw = config.raw["training"]
    distillation = config.raw["distillation"]
    return TrainingSettings(
        batch_size=int(raw["batch_size"]),
        max_epochs=2 if quick else int(raw["max_epochs"]),
        patience=min(2, int(raw["patience"])) if quick else int(raw["patience"]),
        min_delta=float(raw["min_delta"]),
        learning_rate=float(raw["learning_rate"]),
        binn_weight_decay=float(raw["binn_weight_decay"]),
        gradient_clip_norm=float(raw["gradient_clip_norm"]),
        distillation_alpha=float(distillation["alpha"]),
        distillation_temperature=float(distillation["temperature"]),
    )
