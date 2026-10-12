"""Streamlit client for the forecast API. Talks to the API over HTTP only."""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
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
        fig.add_scatter(x=df["ts"], y=df["p90"], line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(
            x=df["ts"], y=df["p10"], fill="tonexty", fillcolor="rgba(0,114,178,0.2)",
            line={"width": 0}, name="80% interval",
        )
        fig.add_scatter(x=df["ts"], y=df["p50"], name="forecast (median)", line={"color": "#0072B2"})
        fig.update_layout(yaxis_title="MW", height=420, margin={"t": 20})
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak (p50)", f"{df['p50'].max():,.0f} MW", f"hour {int(df.loc[df['p50'].idxmax(), 'hour_local'])}")
        c2.metric("Energy (p50)", f"{df['p50'].sum() / 1000:,.1f} GWh")
        c3.metric("Mean interval width", f"{(df['p90'] - df['p10']).mean():,.0f} MW")
        st.caption(
            f"Model {fc['model']} | version {fc['model_version']} | input data up to "
            f"{fc['input_data_timestamp']} | weather: {fc['weather_source']}"
        )
        st.dataframe(df[["ts", "p10", "p50", "p90"]].round(0), use_container_width=True)

with tabs[1]:
    if fc and fc.get("explanation_available"):
        hour = st.slider("Forecast hour index (0-23)", 0, 23, 8)
        ex = safe(get_json, "/v1/explain", hour=hour, **params)
        if ex:
            a = pd.DataFrame(ex["attributions"])
            vals = list(a["attribution_mw"])
            fig = go.Figure(
                go.Waterfall(
                    orientation="h",
                    y=["base value", *a["feature"], "other"],
                    x=[ex["base_value_mw"], *vals, ex["p50"] - ex["base_value_mw"] - sum(vals)],
                    measure=["absolute", *["relative"] * len(vals), "relative"],
                )
            )
            fig.update_layout(height=460, xaxis_title="MW", margin={"t": 20})
            st.plotly_chart(fig, use_container_width=True)
            st.caption(ex["note"])
    else:
        st.info("No explanations available for this forecast.")

with tabs[2]:
    for name, title in (
        ("ladder.csv", "Benchmark ladder (test period)"),
        ("by_hour.csv", "MAE by hour of day"),
        ("ablations.csv", "Ablations"),
    ):
        df = safe(get_csv, name)
        if df is not None:
            st.subheader(title)
            st.dataframe(df, use_container_width=True)
    by_hour = safe(get_csv, "by_hour.csv")
    if by_hour is not None and {"model", "hour", "mae"} <= set(by_hour.columns):
        z = by_hour[by_hour.get("zone", "DE") == zone] if "zone" in by_hour else by_hour
        st.plotly_chart(
            go.Figure([go.Scatter(x=g["hour"], y=g["mae"], name=m) for m, g in z.groupby("model")]),
            use_container_width=True,
        )

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
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.json(dq["zone_sum_vs_national"])
