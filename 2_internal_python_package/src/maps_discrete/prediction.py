from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PredictionBundle:
    hazard_logits: np.ndarray
    log_risk: np.ndarray
    attention: np.ndarray
    hazards: np.ndarray
    survival: np.ndarray
    cumulative_incidence: np.ndarray
    time_group_probability: np.ndarray


def predict(model, omics: np.ndarray, prs: np.ndarray, device: torch.device) -> PredictionBundle:
    model.eval()
    logits_all: list[np.ndarray] = []
    risk_all: list[np.ndarray] = []
    attention_all: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(omics), torch.from_numpy(prs)),
        batch_size=256,
        shuffle=False,
    )
    with torch.no_grad():
        for x_batch, prs_batch in loader:
            output = model(x_batch.to(device), prs_batch.to(device))
            logits_all.append(output["hazard_logits"].cpu().numpy())
            risk_all.append(output["log_risk"].cpu().numpy())
            attention_all.append(output["attention"].cpu().numpy())
    logits = np.concatenate(logits_all).astype(np.float32)
    hazards = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    survival = np.cumprod(1.0 - hazards, axis=1)
    cumulative = 1.0 - survival
    survival_before = np.concatenate(
        [np.ones((len(hazards), 1), dtype=hazards.dtype), survival[:, :-1]], axis=1
    )
    time_probability = np.concatenate([survival_before * hazards, survival[:, -1:]], axis=1)
    return PredictionBundle(
        hazard_logits=logits,
        log_risk=np.concatenate(risk_all).astype(np.float32),
        attention=np.concatenate(attention_all).astype(np.float32),
        hazards=hazards,
        survival=survival,
        cumulative_incidence=cumulative,
        time_group_probability=time_probability,
    )
