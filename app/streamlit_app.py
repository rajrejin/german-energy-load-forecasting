"""Streamlit client for the forecast API. Talks to the API over HTTP only."""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("ENERGY_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("ENERGY_API_KEY")

st.set_page_config(page_title="German demand forecaster", layout="wide")


def api_get(path: str, **params: Any) -> Any:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    r = httpx.get(f"{API_URL}{path}", params=params, headers=headers, timeout=15.0)
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise RuntimeError(f"{r.status_code}: {msg}")
    return r


@st.cache_data(ttl=60, show_spinner=False)
def get_json(path: str, **params: Any) -> Any:
    return api_get(path, **params).json()


@st.cache_data(ttl=300, show_spinner=False)
def get_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(api_get(f"/v1/results/{name}").text))


def safe(fn, *a, **kw):  # type: ignore[no-untyped-def]
    try:
        return fn(*a, **kw)
    except Exception as e:  # network or API error
        st.warning(str(e))
        return None


st.title("German electricity demand forecaster")
st.caption(
    "Day-ahead forecasts per transmission control area. Forecast issue time is 10:00 Berlin "
    "time, the forecast covers the next day. Attributions explain the model, not the physical "
    "causes of demand."
)

zones_resp = safe(get_json, "/v1/zones")
zones = zones_resp["zones"] if zones_resp else []
with st.sidebar:
    st.header("Selection")
    st.write(f"API: `{API_URL}`")
    zone = st.selectbox("Zone", zones or ["DE"])
    use_date = st.checkbox("Pick an issue date (otherwise latest)", value=False)
    issue = st.date_input("Issue date") if use_date else None

tabs = st.tabs(["Forecast", "Explain", "Backtest", "Briefing", "Data quality"])
params: dict[str, Any] = {"zone": zone}
if issue is not None:
    params["issue_time"] = issue.isoformat()
fc = safe(get_json, "/v1/forecast", **params)

with tabs[0]:
    if fc:
        if fc["stale"]:
            st.error(f"This forecast is stale ({fc['age_hours']:.0f} h old).")
        df = pd.DataFrame(fc["points"])
        df["ts"] = pd.to_datetime(df["ts_utc"]).dt.tz_convert("Europe/Berlin")
        fig = go.Figure()
        hist = safe(get_json, "/v1/history", zone=zone)
        if hist:
            h = pd.DataFrame({"ts": pd.to_datetime(hist["ts_utc"]), "load": hist["load_mw"]})
            h["ts"] = h["ts"].dt.tz_convert("Europe/Berlin")
            fig.add_scatter(x=h["ts"], y=h["load"], name="actual (recent)", line={"color": "#555"})
        fig.add_scatter(
            x=df["ts"], y=df["p90"], line={"width": 0}, showlegend=False, hoverinfo="skip"
        )
        fig.add_scatter(
            x=df["ts"],
            y=df["p10"],
            fill="tonexty",
            fillcolor="rgba(0,114,178,0.2)",
            line={"width": 0},
            name="80% interval",
        )
        fig.add_scatter(
            x=df["ts"], y=df["p50"], name="forecast (median)", line={"color": "#0072B2"}
        )
        fig.update_layout(yaxis_title="MW", height=420, margin={"t": 20})
        st.plotly_chart(fig, width="stretch")
        c1, c2, c3 = st.columns(3)
        peak_hour = int(df.loc[df["p50"].idxmax(), "hour_local"])
        c1.metric(f"Peak (p50), local hour {peak_hour}", f"{df['p50'].max():,.0f} MW")
        c2.metric("Energy (p50)", f"{df['p50'].sum() / 1000:,.1f} GWh")
        c3.metric("Mean interval width", f"{(df['p90'] - df['p10']).mean():,.0f} MW")
        st.caption(
            f"Model {fc['model']} | version {fc['model_version']} | input data up to "
            f"{fc['input_data_timestamp']} | weather: {fc['weather_source']}"
        )
        st.dataframe(df[["ts", "p10", "p50", "p90"]].round(0), width="stretch")

with tabs[1]:
    if fc and fc.get("explanation_available"):
        hour = st.slider("Forecast hour index (0-23)", 0, 23, 8)
        ex = safe(get_json, "/v1/explain", hour=hour, **params)
        if ex:
            a = pd.DataFrame(ex["attributions"])
            vals = list(a["attribution_mw"])
            base = float(ex["base_value_mw"])
            steps = np.array([*vals, ex["p50"] - base - sum(vals)])
            fig = go.Figure(
                go.Waterfall(
                    orientation="h",
                    base=base,
                    y=[*a["feature"], "all other features"],
                    x=steps,
                    measure=["relative"] * len(steps),
                )
            )
            # Start the axis at the base value: from zero the attributions are unreadable.
            path = base + np.concatenate([[0.0], np.cumsum(steps)])
            pad = max(0.15 * (path.max() - path.min()), 100.0)
            fig.update_layout(
                height=460,
                xaxis_title=f"MW (starts at the base value {base:,.0f} MW, ends at the forecast)",
                xaxis_range=[path.min() - pad, path.max() + pad],
                yaxis_autorange="reversed",
                margin={"t": 20},
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(ex["note"])
    else:
        st.info("No explanations available for this forecast.")

with tabs[2]:
    regime = st.radio(
        "Weather inputs for the target day",
        ["reanalysis", "archived"],
        horizontal=True,
        help="reanalysis: observed weather as a proxy (optimistic); archived: archived forecasts",
    )
    ladder = safe(get_csv, "ladder.csv")
    if ladder is not None:
        st.subheader(f"Benchmark ladder, {zone}, test period")
        cols = ["model", "n_days", "mae", "mae_ci_lo", "mae_ci_hi", "mape", "mase", "coverage_80"]
        sel = ladder[(ladder["zone"] == zone) & (ladder["regime"] == regime)]
        st.dataframe(sel[[c for c in cols if c in sel]].round(3), width="stretch", hide_index=True)
    by_hour = safe(get_csv, "by_hour.csv")
    if by_hour is not None and {"zone", "regime", "model", "hour", "mae"} <= set(by_hour.columns):
        st.subheader("MAE by hour of the forecast day")
        z = by_hour[(by_hour["zone"] == zone) & (by_hour["regime"] == regime)]
        fig = go.Figure(
            [go.Scatter(x=g["hour"], y=g["mae"], name=m) for m, g in z.groupby("model")]
        )
        fig.update_layout(xaxis_title="hour index (0 = local midnight)", yaxis_title="MAE (MW)")
        st.plotly_chart(fig, width="stretch")
    abl = safe(get_csv, "ablations.csv")
    if abl is not None:
        st.subheader("Ablations (national total, test period)")
        st.dataframe(abl.round(3), width="stretch", hide_index=True)

with tabs[3]:
    b = safe(get_json, "/v1/briefing/latest")
    if b:
        st.subheader(b.get("headline", "Weekly briefing"))
        if b.get("fallback_template"):
            st.warning("Deterministic template (language model output failed verification).")
        for z, bullets in b.get("zones", {}).items():
            st.markdown(f"**{z}**")
            for line in bullets:
                st.markdown(f"- {line}")
        st.markdown("**Risks**")
        for line in b.get("risks", []):
            st.markdown(f"- {line}")
        st.markdown("**Attention items**")
        for line in b.get("attention", []):
            st.markdown(f"- {line}")

with tabs[4]:
    dq = safe(get_json, "/v1/results/data_quality.json")
    if dq:
        rows = [
            {
                "zone": z,
                "rows": v["rows"],
                "interpolated_h": v["load_interpolated_hours"],
                "missing_after_h": v["load_missing_after_treatment_hours"],
                "outlier_h": v["outlier_hours"],
                "last": v["last"],
            }
            for z, v in dq["zones"].items()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.json(dq["zone_sum_vs_national"])
