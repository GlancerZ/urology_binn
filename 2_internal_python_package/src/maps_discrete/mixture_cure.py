"""Discrete-time mixture cure Teacher: onset probability plus onset-year distribution.

The incidence head gives pi(x) = P(onset within the n_intervals-year horizon). The
latency head gives f(t | x, onset) over the same yearly intervals. Training maximises
the censoring-aware mixture likelihood

    onset in year t:              pi * f(t)
    onset-free through c years:   (1 - pi) + pi * (1 - F(c)),   F(c) = f(1) + ... + f(c)

With complete follow-up through the horizon (c = n_intervals) F(c) = 1, so the second
term reduces to 1 - pi and the likelihood separates into incidence BCE on everyone plus
onset-year cross-entropy on cases.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from .modeling import InterpretableRiskFusion, build_model
from .training import FitResult, TrainingSettings

# ordinal: early-onset score from its own token attention; ordinal_risk: score is a
# learned multiple of the incidence logit; ordinal_coupled: both; softmax: free logits.
TIMING_HEADS = ("ordinal", "softmax", "ordinal_risk", "ordinal_coupled")


class OrdinalTimingHead(nn.Module):
    """Cumulative-logit onset-year distribution driven by one early-onset score.

    P(onset in years 1..t | onset) = sigmoid(threshold_t + score), so a larger score
    shifts the whole distribution towards earlier years. The score can come from the
    fused tokens, from the incidence logit (score = coupling * logit), or both.
    """

    def __init__(
        self,
        n_tokens: int,
        n_intervals: int,
        embed_dim: int = 32,
        use_tokens: bool = True,
        use_risk: bool = False,
    ):
        super().__init__()
        if n_intervals < 2:
            raise ValueError("n_intervals must be at least 2")
        if not (use_tokens or use_risk):
            raise ValueError("The early-onset score needs tokens, the incidence logit, or both")
        self.fusion = None
        if use_tokens:
            self.fusion = InterpretableRiskFusion(n_tokens, embed_dim)
            self.fusion.bias.requires_grad_(False)  # location is carried by the thresholds
        self.coupling = nn.Parameter(torch.zeros(1)) if use_risk else None
        self.first_threshold = nn.Parameter(torch.zeros(1))
        self.threshold_steps = nn.Parameter(torch.zeros(n_intervals - 2))

    def thresholds(self) -> torch.Tensor:
        steps = F.softplus(self.threshold_steps)
        return self.first_threshold + torch.cat([steps.new_zeros(1), torch.cumsum(steps, 0)])

    def forward(self, tokens: torch.Tensor, incidence_logit: torch.Tensor | None = None):
        score = tokens.new_zeros(len(tokens))
        attention = None
        if self.fusion is not None:
            score, attention, _ = self.fusion(tokens)
        if self.coupling is not None:
            score = score + self.coupling * incidence_logit
        inner = torch.sigmoid(self.thresholds().unsqueeze(0) + score.unsqueeze(1))
        cdf = torch.cat([inner.new_zeros(len(inner), 1), inner, inner.new_ones(len(inner), 1)], dim=1)
        probability = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1e-12)
        return torch.log(probability), score, attention

    def initialize(self, case_distribution: np.ndarray) -> None:
        cumulative = np.clip(np.cumsum(case_distribution)[:-1], 1e-4, 1.0 - 1e-4)
        thresholds = np.log(cumulative / (1.0 - cumulative))
        steps = np.maximum(np.diff(thresholds), 1e-3)
        with torch.no_grad():
            self.first_threshold.fill_(float(thresholds[0]))
            self.threshold_steps.copy_(torch.from_numpy(np.log(np.expm1(steps))).float())
            if self.fusion is not None:
                self.fusion.bias.zero_()
            if self.coupling is not None:
                self.coupling.zero_()

    def decayed_parameters(self) -> list[nn.Parameter]:
        return []


class SoftmaxTimingHead(nn.Module):
    """Unconstrained onset-year distribution: one logit per year from the fused tokens."""

    def __init__(self, n_tokens: int, n_intervals: int):
        super().__init__()
        self.linear = nn.Linear(n_tokens, n_intervals)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, tokens: torch.Tensor, incidence_logit: torch.Tensor | None = None):
        return torch.log_softmax(self.linear(tokens), dim=1), None, None

    def initialize(self, case_distribution: np.ndarray) -> None:
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.from_numpy(np.log(case_distribution)).float())

    def decayed_parameters(self) -> list[nn.Parameter]:
        return [self.linear.weight]


class MixtureCureBINN(nn.Module):
    """Reactome BINN encoder shared by an incidence head and an onset-year head."""

    def __init__(
        self,
        binn: nn.Module,
        h6_dim: int,
        n_intervals: int,
        attention_dim: int = 32,
        timing_head: str = "ordinal",
    ):
        super().__init__()
        if timing_head not in TIMING_HEADS:
            raise ValueError(f"timing_head must be one of {TIMING_HEADS}, got {timing_head!r}")
        self.binn = binn
        self.h6_bn = nn.BatchNorm1d(h6_dim, eps=1e-5, affine=True)
        self.prs_bn = nn.BatchNorm1d(1, eps=1e-5, affine=True)
        self.incidence = InterpretableRiskFusion(h6_dim + 1, attention_dim)
        if timing_head == "softmax":
            self.timing = SoftmaxTimingHead(h6_dim + 1, n_intervals)
        else:
            self.timing = OrdinalTimingHead(
                h6_dim + 1, n_intervals, attention_dim,
                use_tokens=timing_head in {"ordinal", "ordinal_coupled"},
                use_risk=timing_head in {"ordinal_risk", "ordinal_coupled"},
            )
        self.timing_head = timing_head
        self.n_intervals = n_intervals

    def forward(self, omics: torch.Tensor, prs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        h6 = self.binn.layers.forward_features(omics)
        tokens = torch.cat([self.h6_bn(h6), self.prs_bn(prs)], dim=1)
        incidence_logit, attention, contribution = self.incidence(tokens)
        timing_log_probability, timing_score, timing_attention = self.timing(tokens, incidence_logit)
        return {
            "incidence_logit": incidence_logit,
            "timing_log_probability": timing_log_probability,
            "timing_score": timing_score,
            "attention": attention,
            "contribution": contribution,
            "timing_attention": timing_attention,
            "h6": h6,
        }


def build_mixture_model(
    data_matrix: pd.DataFrame,
    mapping: pd.DataFrame,
    pathways: pd.DataFrame,
    entity_col: str,
    binn_root: str | Path,
    device: torch.device,
    n_layers: int = 6,
    n_intervals: int = 10,
    attention_dim: int = 32,
    timing_head: str = "ordinal",
) -> MixtureCureBINN:
    """Use exactly the frozen Teacher's Reactome BINN encoder (modeling.build_model)."""
    encoder = build_model(
        data_matrix=data_matrix, mapping=mapping, pathways=pathways, entity_col=entity_col,
        binn_root=binn_root, device=torch.device("cpu"), n_layers=n_layers,
        n_intervals=n_intervals, attention_dim=attention_dim,
    ).binn
    h6_dim = int(encoder.layers.blocks[-1][0].out_features)
    return MixtureCureBINN(encoder, h6_dim, n_intervals, attention_dim, timing_head).to(device)


def mixture_nll(
    incidence_logit: torch.Tensor,
    timing_log_probability: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-participant negative log-likelihood of the discrete-time mixture cure model.

    targets is the one-hot onset interval (all zero without onset); for participants
    without onset, mask.sum() is the number of intervals observed onset-free.
    """
    n_intervals = targets.shape[1]
    case = targets.sum(dim=1) > 0
    log_pi = F.logsigmoid(incidence_logit)
    log_not_pi = F.logsigmoid(-incidence_logit)
    case_ll = log_pi + (timing_log_probability * targets).sum(dim=1)
    observed = mask.sum(dim=1)
    later = (torch.arange(n_intervals, device=targets.device).unsqueeze(0) >= observed.unsqueeze(1)).float()
    latency_survival = (timing_log_probability.exp() * later).sum(dim=1)
    censored_ll = torch.where(
        observed >= n_intervals,
        log_not_pi,
        torch.logaddexp(log_not_pi, log_pi + torch.log(latency_survival.clamp_min(1e-12))),
    )
    return -torch.where(case, case_ll, censored_ll)


def mixture_validation_nll(
    model: MixtureCureBINN,
    omics: np.ndarray,
    prs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    index: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    loader = DataLoader(TensorDataset(
        torch.from_numpy(omics[index]),
        torch.from_numpy(prs[index]),
        torch.from_numpy(targets[index]),
        torch.from_numpy(mask[index]),
    ), batch_size=256, shuffle=False)
    with torch.no_grad():
        for x_batch, prs_batch, y_batch, mask_batch in loader:
            output = model(x_batch.to(device), prs_batch.to(device))
            nll = mixture_nll(
                output["incidence_logit"], output["timing_log_probability"],
                y_batch.to(device), mask_batch.to(device),
            )
            total += float(nll.sum())
            count += len(nll)
    return total / max(count, 1)


def case_interval_distribution(targets: np.ndarray, index: np.ndarray, smoothing: float = 0.5) -> np.ndarray:
    """Smoothed onset-year distribution among cases in index (population reference)."""
    counts = targets[index].sum(axis=0).astype(np.float64) + smoothing
    return counts / counts.sum()


def initialize_mixture_heads(model: MixtureCureBINN, targets: np.ndarray, train_index: np.ndarray) -> None:
    case = targets[train_index].sum(axis=1) > 0
    rate = (case.sum() + 0.5) / (len(case) + 1.0)
    with torch.no_grad():
        model.incidence.bias.fill_(float(np.log(rate / (1.0 - rate))))
    model.timing.initialize(case_interval_distribution(targets, train_index))


def _mixture_optimizer(model: MixtureCureBINN, settings: TrainingSettings) -> Adam:
    decayed = model.timing.decayed_parameters()
    decayed_ids = {id(p) for p in decayed}
    heads = [
        p for name, p in model.named_parameters()
        if not name.startswith("binn.") and p.requires_grad and id(p) not in decayed_ids
    ]
    groups = [
        {
            "params": model.binn.parameters(),
            "lr": settings.learning_rate,
            "weight_decay": settings.binn_weight_decay,
        },
        {"params": heads, "lr": settings.learning_rate, "weight_decay": 0.0},
    ]
    if decayed:
        groups.append({
            "params": decayed,
            "lr": settings.learning_rate,
            "weight_decay": settings.binn_weight_decay,
        })
    return Adam(groups)


def fit_mixture_model(
    model: MixtureCureBINN,
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
) -> FitResult:
    """Fit with the frozen Teacher's optimiser, batching and validation-NLL early stopping."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # settings.min_delta is defined on the frozen Teacher's per person-interval NLL; the
    # mixture NLL is per participant, so rescale by the mean number of observed intervals.
    min_delta = settings.min_delta * float(mask[validation_index].sum(axis=1).mean())
    initialize_mixture_heads(model, targets, train_index)
    optimizer = _mixture_optimizer(model, settings)
    dataset = TensorDataset(
        torch.from_numpy(omics[train_index]),
        torch.from_numpy(prs[train_index]),
        torch.from_numpy(targets[train_index]),
        torch.from_numpy(mask[train_index]),
    )
    best_loss = float("inf")
    best_epoch = 0
    waiting = 0
    best_state = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        losses: list[float] = []
        generator = torch.Generator().manual_seed(int(training_seed) + epoch)
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True, generator=generator)
        for x_batch, prs_batch, y_batch, mask_batch in loader:
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(x_batch.to(device), prs_batch.to(device))
            loss = mixture_nll(
                output_batch["incidence_logit"], output_batch["timing_log_probability"], y_batch, mask_batch,
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = mixture_validation_nll(model, omics, prs, targets, mask, validation_index, device)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "validation_nll": val_loss})
        print(
            f"{model_id} seed={training_seed} epoch={epoch:03d} "
            f"train={np.mean(losses):.6f} val_nll={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss - min_delta:
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


@dataclass
class MixturePrediction:
    incidence_probability: np.ndarray
    onset_year_probability: np.ndarray
    cumulative_incidence: np.ndarray
    timing_score: np.ndarray | None
    attention: np.ndarray
    timing_attention: np.ndarray | None


def predict_mixture(model: MixtureCureBINN, omics: np.ndarray, prs: np.ndarray, device: torch.device) -> MixturePrediction:
    model.eval()
    parts: dict[str, list[np.ndarray]] = {
        "pi": [], "onset": [], "score": [], "attention": [], "timing_attention": [],
    }
    loader = DataLoader(
        TensorDataset(torch.from_numpy(omics), torch.from_numpy(prs)),
        batch_size=256,
        shuffle=False,
    )
    with torch.no_grad():
        for x_batch, prs_batch in loader:
            output = model(x_batch.to(device), prs_batch.to(device))
            parts["pi"].append(torch.sigmoid(output["incidence_logit"]).cpu().numpy())
            parts["onset"].append(output["timing_log_probability"].exp().cpu().numpy())
            parts["attention"].append(output["attention"].cpu().numpy())
            if output["timing_score"] is not None:
                parts["score"].append(output["timing_score"].cpu().numpy())
            if output["timing_attention"] is not None:
                parts["timing_attention"].append(output["timing_attention"].cpu().numpy())
    pi = np.concatenate(parts["pi"]).astype(np.float64)
    onset = np.concatenate(parts["onset"]).astype(np.float64)
    onset /= onset.sum(axis=1, keepdims=True)
    return MixturePrediction(
        incidence_probability=pi,
        onset_year_probability=onset,
        cumulative_incidence=pi[:, None] * np.cumsum(onset, axis=1),
        timing_score=np.concatenate(parts["score"]) if parts["score"] else None,
        attention=np.concatenate(parts["attention"]).astype(np.float32),
        timing_attention=np.concatenate(parts["timing_attention"]).astype(np.float32) if parts["timing_attention"] else None,
    )
