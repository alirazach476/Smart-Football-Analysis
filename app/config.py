"""
Application configuration via environment variables.

Settings are loaded from the environment (and an optional `.env` file) with
sensible defaults so the API runs out of the box for local development while
remaining configurable for production deployments.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:  # pydantic-settings is the v2 home for BaseSettings
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - fallback if dependency missing
    _HAS_PYDANTIC_SETTINGS = False


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_FIELD_MODEL = _PROJECT_ROOT / "football-field-detection" / "football-field-detection.pt"
_DEFAULT_OBJECT_MODEL = _PROJECT_ROOT / "football-players-detection" / "football-players-detection.pt"
_DEFAULT_BALL_MODEL = _PROJECT_ROOT / "football-ball-detection" / "football-ball-detection.pt"


if _HAS_PYDANTIC_SETTINGS:

    class Settings(BaseSettings):
        """Runtime configuration."""

        model_config = SettingsConfigDict(
            env_prefix="SFA_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # --- API metadata --------------------------------------------------
        app_name: str = "Smart Football Match Analysis System"
        api_prefix: str = "/api/v1"
        debug: bool = False

        # --- Storage -------------------------------------------------------
        data_dir: Path = _PROJECT_ROOT / "data"
        database_url: str = ""  # derived from data_dir if empty

        # --- Models --------------------------------------------------------
        field_model_path: Path = _DEFAULT_FIELD_MODEL
        object_model_path: Path = _DEFAULT_OBJECT_MODEL

        # --- Processing ----------------------------------------------------
        max_workers: int = 1
        device: str = "cpu"
        imgsz: int = 1280
        conf_threshold: float = 0.25
        ball_conf_threshold: float = 0.08
        possession_radius_m: float = 5.5
        enable_ball_model: bool = True
        ball_model: str = str(_DEFAULT_BALL_MODEL)
        position_smoothing: float = 0.6
        enable_heatmaps_default: bool = False

        # --- Uploads -------------------------------------------------------
        max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB
        allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")

        # --- CORS ----------------------------------------------------------
        cors_origins: tuple[str, ...] = ("*",)

        @property
        def jobs_dir(self) -> Path:
            return self.data_dir / "jobs"

        @property
        def sqlalchemy_url(self) -> str:
            if self.database_url:
                return self.database_url
            return f"sqlite:///{(self.data_dir / 'app.db').as_posix()}"

        def ensure_dirs(self) -> None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.jobs_dir.mkdir(parents=True, exist_ok=True)

else:  # pragma: no cover - minimal fallback without pydantic-settings

    class Settings:  # type: ignore[no-redef]
        """Environment-driven settings (fallback implementation)."""

        def __init__(self) -> None:
            self.app_name = os.getenv("SFA_APP_NAME", "Smart Football Match Analysis System")
            self.api_prefix = os.getenv("SFA_API_PREFIX", "/api/v1")
            self.debug = os.getenv("SFA_DEBUG", "false").lower() == "true"
            self.data_dir = Path(os.getenv("SFA_DATA_DIR", str(_PROJECT_ROOT / "data")))
            self.database_url = os.getenv("SFA_DATABASE_URL", "")
            self.field_model_path = Path(os.getenv("SFA_FIELD_MODEL_PATH", str(_DEFAULT_FIELD_MODEL)))
            self.object_model_path = Path(os.getenv("SFA_OBJECT_MODEL_PATH", str(_DEFAULT_OBJECT_MODEL)))
            self.max_workers = int(os.getenv("SFA_MAX_WORKERS", "1"))
            self.device = os.getenv("SFA_DEVICE", "cpu")
            self.imgsz = int(os.getenv("SFA_IMGSZ", "1280"))
            self.conf_threshold = float(os.getenv("SFA_CONF_THRESHOLD", "0.25"))
            self.ball_conf_threshold = float(os.getenv("SFA_BALL_CONF_THRESHOLD", "0.08"))
            self.possession_radius_m = float(os.getenv("SFA_POSSESSION_RADIUS_M", "5.5"))
            self.enable_ball_model = os.getenv("SFA_ENABLE_BALL_MODEL", "true").lower() == "true"
            self.ball_model = os.getenv("SFA_BALL_MODEL", str(_DEFAULT_BALL_MODEL))
            self.position_smoothing = float(os.getenv("SFA_POSITION_SMOOTHING", "0.6"))
            self.enable_heatmaps_default = os.getenv("SFA_ENABLE_HEATMAPS_DEFAULT", "false").lower() == "true"
            self.max_upload_bytes = int(os.getenv("SFA_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
            self.allowed_video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
            self.cors_origins = tuple(
                o for o in os.getenv("SFA_CORS_ORIGINS", "*").split(",") if o
            )

        @property
        def jobs_dir(self) -> Path:
            return self.data_dir / "jobs"

        @property
        def sqlalchemy_url(self) -> str:
            if self.database_url:
                return self.database_url
            return f"sqlite:///{(self.data_dir / 'app.db').as_posix()}"

        def ensure_dirs(self) -> None:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.jobs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> "Settings":
    settings = Settings()
    settings.ensure_dirs()
    return settings
