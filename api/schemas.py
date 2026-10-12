"""Response schemas for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ForecastPoint(BaseModel):
    ts_utc: str
    hour_local: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    zone: str
    issue_time: str
    issue_time_local: str
    target_date: str
    model: str
    model_version: str
    input_data_timestamp: str
    weather_source: str
    calibration_offset_mw: float
    points: list[ForecastPoint] = Field(min_length=24, max_length=24)
    explanation_available: bool
    created_utc: str
    stale: bool
    age_hours: float


class Attribution(BaseModel):
    feature: str
    attribution_mw: float


class ExplainResponse(BaseModel):
    zone: str
    target_date: str
    hour_index: int
    ts_utc: str
    p50: float
    base_value_mw: float
    attributions: list[Attribution]
    note: str


class ZonesResponse(BaseModel):
    zones: list[str]
    description: dict[str, str]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    detail: dict[str, Any]
