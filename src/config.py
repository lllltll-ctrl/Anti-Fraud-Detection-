from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_dir", Path(self.base_dir))

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def submissions_dir(self) -> Path:
        return self.artifacts_dir / "submissions"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    @property
    def processed_dir(self) -> Path:
        return self.artifacts_dir / "processed"

    def ensure_directories(self) -> None:
        for path in [self.artifacts_dir, self.models_dir, self.submissions_dir, self.reports_dir, self.processed_dir]:
            path.mkdir(parents=True, exist_ok=True)
