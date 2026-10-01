"""Internal utilities for reproducible discrete-time MAPS experiments."""

from .config import AnalysisConfig, load_config
from .guard import ProtocolNotApprovedError, require_approved
from .seeding import set_global_seed
from .splits import SplitAudit, audit_split_manifest
from .metrics import (
    bootstrap_model_comparison,
    bootstrap_model_performance,
    continuous_nri_idi,
    decision_curve,
    evaluate_model,
    harrell_c_index,
)

__all__ = [
    "AnalysisConfig",
    "ProtocolNotApprovedError",
    "SplitAudit",
    "audit_split_manifest",
    "bootstrap_model_comparison",
    "bootstrap_model_performance",
    "continuous_nri_idi",
    "decision_curve",
    "evaluate_model",
    "harrell_c_index",
    "load_config",
    "require_approved",
    "set_global_seed",
]

__version__ = "0.2.0"
