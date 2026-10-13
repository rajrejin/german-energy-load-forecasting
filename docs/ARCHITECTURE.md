# Architecture

This matches what is implemented in `src/energy`, `api` and `app`.

## Data and training flow

```mermaid
flowchart LR
  SMARD[SMARD chart_data API<br/>load 410, forecast 411, generation] -->|energy fetch| RAW[(data/raw<br/>gzip JSON cache)]
  OM[Open-Meteo<br/>archive + historical forecast] -->|energy fetch| RAW
  RAW -->|energy build-dataset| TAB[(data/processed<br/>hourly UTC table per zone)]
  TAB --> Q[quality.py<br/>data_quality.json]
  TAB --> EDA[eda.py<br/>figures]
  TAB --> ORI[features.py + windows.py<br/>origins at 10:00 local, 24 h block]
  ORI --> M1[naive]
  ORI --> M2[MSTL-ARIMA]
  ORI --> M3[LightGBM]
  ORI --> M4[LSTM per zone / pooled]
  M3 --> ART[(artifacts/models<br/>manifest.json per model)]
  M4 --> ART
  ART -->|energy backtest| PRED[(preds npz<br/>val/test, 2 weather regimes)]
  M1 --> PRED
  M2 --> PRED
  PRED -->|energy report| RES[(results/*.csv, json)]
  ART -->|energy explain| SHAP[SHAP GradientExplainer<br/>explain_sanity.json]
```

## Serving flow

```mermaid
flowchart LR
  SM[SMARD latest load] --> R[energy refresh]
  LW[Open-Meteo live forecast] --> R
  ST[(ArtifactStore<br/>local / Azurite / Azure Blob)] -->|models + manifest| R
  R -->|input checks pass| ST2[(forecasts/zone/date.json<br/>explanations, actuals)]
  R -->|checks fail| STAT[status/refresh.json<br/>last good forecast kept]
  ST2 --> API[FastAPI<br/>/v1/forecast /v1/explain ...]
  STAT --> API
  API --> APP[Streamlit app<br/>HTTP only]
  ST2 --> B[energy briefing<br/>LLM + numeric verification]
  B --> ST2
```

## Modules

| Area | Modules |
| --- | --- |
| Data | `data/smard.py`, `data/smard_csv.py`, `data/openmeteo.py`, `data/build.py`, `data/quality.py`, `data/eda.py`, `calendar_features.py` |
| Samples | `windows.py` (origins and 24 h blocks), `features.py` (features with provenance), `splits.py` |
| Models | `models/naive.py`, `stat_mstl.py`, `gbm.py`, `lstm.py`, `pooled_lstm.py`, all with `fit`, `predict`, `predict_quantiles` |
| Training | `training/train.py`, `hpo.py`, `parallel.py` (Dask), `ablations.py`, `manifest.py` |
| Evaluation | `evaluation/metrics.py`, `backtest.py`, `runner.py`, `intervals.py` (conformal calibration), `significance.py`, `analysis.py` |
| Explainability | `explain/shap_lstm.py`, `sanity.py`, `plots.py` |
| Serving | `store/*`, `serving/live.py`, `refresh.py`, `forecast_service.py`, `api/*`, `app/streamlit_app.py` |
| Briefing | `briefing/aggregate.py`, `llm.py`, `verify.py`, `evaluate.py` |

## Key invariants

* Everything is stored in UTC; calendar features use Europe/Berlin local time of each hour.
* A forecast origin reads load and observed weather only at or before the last complete hour before issue time (tested by provenance checks and by perturbing the future).
* Scaling statistics are fit on the training period only (stored in the LSTM artifact).
* The API refuses artifacts whose manifest is missing, incomplete or inconsistent with the file hashes (the refresh job validates them before forecasting; the API only serves what the refresh job published).
