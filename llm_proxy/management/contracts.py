from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateResult:
    valid: bool
    revision: str | None
    restart_required: bool
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "revision": self.revision, "restart_required": self.restart_required, "errors": list(self.errors)}
