"""Read side of the serving layer: forecasts, explanations and briefings from the store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from energy.config import ALL_REGIONS
from energy.store.base import ArtifactStore, KeyNotFoundError


class ForecastNotFound(Exception):
    pass


@dataclass
class ForecastService:
    store: ArtifactStore
    stale_after_hours: float = 36.0

    def zones(self) -> list[str]:
        return [z for z in ALL_REGIONS if self.store.exists(f"forecasts/{z}/latest.json")]

    def _stale(self, fc: dict[str, Any], now: pd.Timestamp | None = None) -> tuple[bool, float]:
        now = now or pd.Timestamp.now(tz="UTC")
        age_h = (now - pd.Timestamp(fc["issue_time"])).total_seconds() / 3600.0
        # a forecast covers the next local day; it is stale when its issue time is old
        return age_h > self.stale_after_hours, age_h

    def forecast(self, zone: str, target_date: str | None = None) -> dict[str, Any]:
        key = f"forecasts/{zone}/{target_date or 'latest'}.json"
        try:
            fc: dict[str, Any] = self.store.get_json(key)
        except KeyNotFoundError as e:
            raise ForecastNotFound(f"no forecast for {zone} {target_date or 'latest'}") from e
        stale, age = self._stale(fc)
        fc = {k: v for k, v in fc.items() if k != "explanations"}
        fc["stale"] = stale
        fc["age_hours"] = round(age, 2)
        return fc

    def explain(self, zone: str, hour: int, target_date: str | None = None) -> dict[str, Any]:
        key = f"forecasts/{zone}/{target_date or 'latest'}.json"
        try:
            fc: dict[str, Any] = self.store.get_json(key)
        except KeyNotFoundError as e:
            raise ForecastNotFound(f"no forecast for {zone}") from e
        if not fc.get("explanation_available"):
            raise ForecastNotFound("explanations were not computed for this forecast")
        pt = fc["points"][hour]
        items = fc["explanations"][str(hour)]
        base = next(i["attribution_mw"] for i in items if i["feature"] == "_base_value")
        return {
            "zone": zone,
            "target_date": fc["target_date"],
            "hour_index": hour,
            "ts_utc": pt["ts_utc"],
            "p50": pt["p50"],
            "base_value_mw": base,
            "attributions": [i for i in items if i["feature"] != "_base_value"],
            "note": "Attributions explain the model, not the physical causes of demand.",
        }

    def history(self, zone: str) -> dict[str, Any]:
        try:
            h: dict[str, Any] = self.store.get_json(f"actuals/{zone}.json")
            return h
        except KeyNotFoundError as e:
            raise ForecastNotFound(f"no history for {zone}") from e

    def result_file(self, name: str) -> bytes:
        try:
            return self.store.get_bytes(f"results/{name}")
        except KeyNotFoundError as e:
            raise ForecastNotFound(f"result {name} not published") from e

    def briefing(self) -> dict[str, Any]:
        try:
            b: dict[str, Any] = self.store.get_json("briefing/latest.json")
            return b
        except KeyNotFoundError as e:
            raise ForecastNotFound("no briefing published yet") from e

    def refresh_status(self) -> dict[str, Any] | None:
        try:
            s: dict[str, Any] = self.store.get_json("status/refresh.json")
            return s
        except KeyNotFoundError:
            return None

    def ready(self) -> dict[str, Any]:
        zs = self.zones()
        ages = {}
        for z in zs:
            fc = self.store.get_json(f"forecasts/{z}/latest.json")
            ages[z] = round(self._stale(fc)[1], 2)
        status = self.refresh_status()
        return {
            "artifacts_loaded": bool(zs),
            "zones": zs,
            "latest_forecast_age_hours": ages,
            "last_refresh": status["run_utc"] if status else None,
            "last_refresh_published": status["published"] if status else None,
        }
