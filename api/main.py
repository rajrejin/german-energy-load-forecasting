"""FastAPI service. Reads published forecasts from the artifact store (no model code at runtime)."""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
import time
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import (
    ErrorResponse,
    ExplainResponse,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
    ZonesResponse,
)
from api.settings import ApiSettings
from energy.config import ALL_REGIONS
from energy.logging_utils import request_id_var, setup_logging
from energy.serving.forecast_service import ForecastNotFound, ForecastService
from energy.serving.settings import StoreSettings, store_from_env
from energy.store.base import ArtifactStore

log = logging.getLogger("energy.api")

REQUESTS = Counter("energy_api_requests_total", "Requests", ["path", "method", "status"])
LATENCY = Histogram("energy_api_request_seconds", "Request latency", ["path"])
FORECAST_AGE = Gauge("energy_forecast_age_hours", "Age of the latest forecast", ["zone"])

ZONE_DESC = {
    "DE": "Germany total (direct national model)",
    "50Hertz": "50Hertz control area",
    "Amprion": "Amprion control area",
    "TenneT": "TenneT control area",
    "TransnetBW": "TransnetBW control area",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def create_app(store: ArtifactStore | None = None, settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    setup_logging(json_logs=settings.json_logs)
    store = store or store_from_env()
    service = ForecastService(store, StoreSettings().stale_after_hours)
    app = FastAPI(title="German electricity demand forecaster", version="0.1.0")
    app.state.service = service

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
            allow_methods=["GET"],
            allow_headers=["X-API-Key"],
        )

    def err(status: int, code: str, message: str) -> JSONResponse:
        body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
        return JSONResponse(status_code=status, content=body)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Any]) -> Any:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            dt = time.perf_counter() - t0
            path = request.url.path
            REQUESTS.labels(path, request.method, str(status)).inc()
            LATENCY.labels(path).observe(dt)
            log.info("%s %s -> %s in %.1f ms", request.method, path, status, dt * 1000)
            request_id_var.reset(token)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return err(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", [])[1:])
        return err(422, "invalid_parameter", f"{loc}: {first.get('msg', 'invalid request')}")

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return err(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return err(500, "internal_error", "internal error")  # no stack trace in the response

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if not settings.api_key:  # unset or empty: authentication disabled (local use only)
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise ApiError(401, "unauthorized", "missing or invalid API key")

    async def run(fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(partial(fn, *a, **kw)), timeout=settings.request_timeout_s
            )
        except TimeoutError as e:
            raise ApiError(504, "timeout", "the artifact store did not answer in time") from e
        except ForecastNotFound as e:
            raise ApiError(404, "not_found", str(e)) from e

    def check_zone(zone: str) -> str:
        if zone not in ALL_REGIONS:
            raise ApiError(422, "invalid_zone", f"zone must be one of {ALL_REGIONS}")
        return zone

    def target_from_issue(issue_time: str | None) -> str | None:
        """Issue time is a date or an ISO datetime; the forecast covers the next local day."""
        if issue_time is None:
            return None
        try:
            ts = pd.Timestamp(issue_time)
        except (ValueError, TypeError) as e:
            raise ApiError(
                422, "invalid_issue_time", "issue_time must be an ISO date or datetime"
            ) from e
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Berlin")
        now = pd.Timestamp.now(tz="UTC")
        if ts > now + pd.Timedelta(days=1):
            raise ApiError(422, "invalid_issue_time", "issue_time lies in the future")
        if ts < now - pd.Timedelta(days=settings.max_forecast_age_days):
            raise ApiError(422, "invalid_issue_time", "issue_time is too far in the past")
        local = ts.tz_convert("Europe/Berlin").normalize().tz_localize(None)
        return (local + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    @app.get("/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/v1/ready",
        response_model=ReadyResponse,
        responses={503: {"model": ErrorResponse}},
        dependencies=[Depends(require_key)],
    )
    async def ready() -> Any:
        detail = await run(service.ready)
        ok = bool(detail["artifacts_loaded"])
        for z, age in detail["latest_forecast_age_hours"].items():
            FORECAST_AGE.labels(z).set(age)
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "detail": detail})

    @app.get("/v1/zones", response_model=ZonesResponse, dependencies=[Depends(require_key)])
    async def zones() -> ZonesResponse:
        zs = await run(service.zones)
        return ZonesResponse(zones=zs, description={z: ZONE_DESC[z] for z in zs})

    @app.get(
        "/v1/forecast",
        response_model=ForecastResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        dependencies=[Depends(require_key)],
    )
    async def forecast(
        zone: str = Query(..., description="DE, 50Hertz, Amprion, TenneT or TransnetBW"),
        issue_time: str | None = Query(None, description="ISO date or datetime; default latest"),
    ) -> dict[str, Any]:
        check_zone(zone)
        target = target_from_issue(issue_time)
        fc: dict[str, Any] = await run(service.forecast, zone, target)
        return fc

    @app.get(
        "/v1/explain",
        response_model=ExplainResponse,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        dependencies=[Depends(require_key)],
    )
    async def explain(
        zone: str = Query(...),
        hour: int = Query(..., ge=0, le=23, description="hour index 0-23 of the forecast day"),
        issue_time: str | None = Query(None),
    ) -> dict[str, Any]:
        check_zone(zone)
        target = target_from_issue(issue_time)
        out: dict[str, Any] = await run(service.explain, zone, hour, target)
        return out

    @app.get(
        "/v1/history",
        responses={404: {"model": ErrorResponse}},
        dependencies=[Depends(require_key)],
    )
    async def history(zone: str = Query(...)) -> dict[str, Any]:
        check_zone(zone)
        h: dict[str, Any] = await run(service.history, zone)
        return h

    @app.get(
        "/v1/results/{name}",
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        dependencies=[Depends(require_key)],
    )
    async def results(name: str) -> Any:
        """Published result tables (JSON or CSV) used by the dashboard."""
        if not re.fullmatch(r"[a-z0-9_]{1,60}\.(json|csv)", name):
            raise ApiError(422, "invalid_name", "name must match [a-z0-9_]+.(json|csv)")
        raw: bytes = await run(service.result_file, name)
        media = "application/json" if name.endswith(".json") else "text/csv"
        return Response(content=raw, media_type=media)

    @app.get("/v1/briefing/latest", dependencies=[Depends(require_key)])
    async def briefing() -> dict[str, Any]:
        b: dict[str, Any] = await run(service.briefing)
        return b

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_key)])
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)

    return app


def get_app() -> FastAPI:  # pragma: no cover - `uvicorn api.main:get_app --factory`
    return create_app()
