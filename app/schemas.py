from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import date


class FundingRequest(BaseModel):
    founded_date: date = Field(..., description="Company founded date")
    number_of_founders: int = Field(ge=0, default=2)
    number_of_investors: int = Field(ge=0, default=5)
    number_of_funding_rounds: int = Field(ge=0, default=3)
    patents_granted: float = Field(ge=0, default=0)

    # Buckets (server computes numeric bounds from label you send OR send bounds directly)
    employees_label: str
    revenue_label: str

    # Single-label categoricals
    headquarters_location: str
    last_funding_type: str

    # Multi-select
    industries: List[str] = []

    @field_validator("industries", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class FundingResponse(BaseModel):
    predicted_total_funding_usd: float
    log_transform: bool
    safe_target: str | None
    safe_target_log: str | None
    n_features_fed: int