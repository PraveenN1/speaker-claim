from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    gemini_model: str

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", ".")),
            database_url=os.environ.get("DATABASE_URL", "sqlite:///speaker_claims.db"),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        )
