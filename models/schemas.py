from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Preferences(BaseModel):
    job_title: str = ""
    location: str = "Remote"
    remote_only: bool = True
    experience_level: str = "Mid"
    employment_type: list[str] = Field(default_factory=lambda: ["Full-time"])
    max_applications: int = 10


class RunCreateRequest(BaseModel):
    preferences: dict[str, Any]
    dry_run: bool = True


class RunCreateResponse(BaseModel):
    run_id: str


class ProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    profile: dict[str, Any]


class RunSummary(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    summary: dict[str, Any]


class StepDTO(BaseModel):
    id: int
    run_id: str
    timestamp: datetime
    agent: str
    event_type: str
    data: dict[str, Any]
    status: str