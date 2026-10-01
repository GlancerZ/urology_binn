from __future__ import annotations

from .config import AnalysisConfig


class ProtocolNotApprovedError(RuntimeError):
    """Raised when a formal run is attempted before protocol approval."""


def require_approved(config: AnalysisConfig) -> None:
    if not config.approved:
        raise ProtocolNotApprovedError(
            "Analysis protocol is still draft. Review OPEN_DECISIONS.md and set "
            "protocol.approved=true only after written approval."
        )

