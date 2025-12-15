from __future__ import annotations
from pathlib import Path
import os, json
from typing import Mapping


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # ---------- CORS ----------
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

    # ---------- Raw training schema names ----------
    FOUNDED_COL = "Founded Date"
    IND_COL = "Industries"

    # =========================
    # Model selection (mapping)
    # =========================
    MODEL_TYPE = os.getenv("MODEL_TYPE", "combined").lower()
    MODEL_DIR_MAP: dict[str, Path] = {
        "combined": Path("models_combined"),
        "city": Path("models_city"),
        "state": Path("models_state"),
        "money": Path("models_rev_split"),
    }


    @classmethod
    def resolve_model_dir(cls, model_type: str | None = None) -> Path:
        key = (model_type or cls.MODEL_TYPE or "combined").lower()
        return cls.MODEL_DIR_MAP.get(key, cls.MODEL_DIR_MAP["combined"])

    # ============================
    # Artifacts selection (mapping)
    # ============================
    ARTIFACTS_DIR_MAP: dict[str, Path] = {
        "combined": Path("artifacts_combined"),
        "city": Path("artifacts_city"),
        "state": Path("artifacts_state"),
        "money": Path("artifacts_rev_split"),
    }


    @classmethod
    def resolve_artifacts_dir(cls, artifacts_type: str | None = None) -> Path:
        key = (artifacts_type or cls.MODEL_TYPE or "combined").lower()
        return cls.ARTIFACTS_DIR_MAP.get(key, cls.ARTIFACTS_DIR_MAP["combined"])
