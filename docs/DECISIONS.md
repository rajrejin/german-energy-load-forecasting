# Decisions

Each entry: decision, alternatives considered, reason. Dates are 2026-10-12 unless stated.

## Data

**D1. SMARD access path: the chunk endpoint (`/app/chart_data/...`), with the CSV export as a verified second parser.**
Alternatives: CSV export only. Reason: the chunk endpoint answered for every needed filter and region at run time (index of 616 weekly chunks per series, data from 2015-01-01 to the day of the run). The CSV importer (`smard_csv.py`) was also built and checked against a real export obtained from the same download endpoint the website uses; on the 168 shared hours of a test week the two parsers agree to 0.01 MW (test `test_csv_and_json_parsers_agree_on_real_data`).

**D2. Filters and regions.** All five regions (`DE`, `50Hertz`, `Amprion`, `TenneT`, `TransnetBW`) returned data for filter 410 (grid load). Filter 411 returned the published day-ahead consumption forecast for all five regions and is used as a labelled external benchmark only. `results/data_availability.csv` records, per filter and region, row counts and first and last timestamps; combinations with no data (for example lignite for TransnetBW, which returned HTTP 404) are listed there.

**D3. Hourly resolution is used as returned.** No quarter-hour aggregation was needed (the hourly series had no missing values). The aggregation rule is implemented (mean power, hours with fewer than four quarter-hour values dropped) and unit tested on a hand-computed case.

**D4. Data start is 2015-01-01 00:00 UTC.** SMARD's first hour is 2014-12-31 23:00 UTC (00:00 local); that single hour is dropped so the table starts on a whole UTC day.

**D5. Weather regimes.** Training and the main backtest use the Open-Meteo reanalysis archive (ERA5 based) as a proxy for forecasts. The realistic regime uses the Historical Forecast API, which in practice returned data from 2021-01-01. That API stitches the first hours of successive model runs, so its lead time is shorter than a true 24 to 48 hour ahead forecast; the realistic regime is therefore still somewhat optimistic. The previous-runs API (forecasts issued a given number of days before) would be closer to day-ahead and is listed as future work in KNOWN_ISSUES.

**D6. Spatial aggregation.** Population-weighted means over 3 to 5 cities per control area (weights in `configs/zones.yaml`, approximate city populations). National weather is the mean of zone weather weighted by training-period mean zone load (`results/national_weights.json`). Holiday shares per zone use approximate state population shares. All approximate and stated as a limitation.

**D7. Short gaps.** Linear interpolation for gaps up to 3 hours, flagged. In this dataset the load series had no gaps at all (see `results/data_quality.json`), so the rule was exercised only in unit tests on hand-written inputs.

## Task definition

**D8. Forecast origin and target block.** Issue time is 10:00 Europe/Berlin on day D. The last observed load hour is the one starting at 09:00 local (complete at issue). The 24 target hours are the 24 consecutive UTC hours starting at local midnight of D+1. On a 23 hour local day the block ends one hour into D+2; on a 25 hour day it stops one hour before the end of D+1. Blocks whose last hour falls outside a split are dropped, so targets never cross a split boundary. Alternative: one row per local clock hour (24 slots, interpolating the missing hour). Reason: the UTC block keeps every target a real, observed hour and keeps lags exact; calendar features of each hour are computed from its true local time.

**D9. Publication latency of load data is assumed to be zero.** At issue time the load of the hour that ended at issue is treated as known. Real SMARD latency is larger, so this is optimistic; the lag structure (the model mostly uses data that is at least 15 hours old for the first target hour) limits the effect.

**D10. Lags.** `lag48`, `lag72`, `lag168`, `lag336` relative to the target hour are always observed at issue time. `lag24` is observed only for target hours up to 09:00 and is NaN otherwise (LightGBM handles NaN, the LSTM does not use it). `latest_same_hour` is the same clock hour 24 h earlier if observed, else 48 h earlier. The previous-day naive variant uses this definition, which is why it is a weak baseline across weekends.

**D11. Splits.** Train 2015-01-01 to 2022-12-31, validation 2023, test 2024-01-01 to the last complete day. Defined by local target date.

**D12. LSTM training augmentation.** With one issue time per day there are about 2900 training origins per zone. The LSTM trains on origins shifted by 0, 6, 12 and 18 hours (issue time and target block shifted together), about 11,600 origins; evaluation always uses the 10:00 issue. The hour-of-day embedding of the target hours lets the model handle the shifted phase. The hyperparameter search uses 0 and 12 hour shifts only, 10 epochs and patience 3 (a cheaper protocol, because a full-protocol search was estimated at several hours on this laptop).

**D13. Hyperparameter search scope.** Random search (20 trials, seed 0) on validation, run on the national series `DE` only, then applied to all zones (per-zone searches would multiply the cost by five). LightGBM is selected by validation MAE, the LSTM by validation pinball loss. Logs are in `results/hpo/`.

**D14. Relative targets for LightGBM.** Load lags and the target are divided by the trailing weekly mean load, so trees do not have to extrapolate over level shifts (demand trend, 2022 crisis). The level itself remains a feature.

**D15. Probabilistic output and calibration.** LSTM: quantiles 0.1, 0.5, 0.9 from one network, non-crossing by construction (median plus and minus softplus offsets). LightGBM: separate quantile objectives for 0.1 and 0.9. Naive models: empirical residual quantiles per horizon from validation. All 80 percent intervals are then widened or narrowed by one constant, fit on validation only (split conformalised quantile regression, Romano et al. 2019), separately for each weather regime. Raw (uncalibrated) coverage is reported next to calibrated coverage.

**D16. MSTL-ARIMA.** statsforecast `MSTL` with seasonal lengths 24 and 168 and an `AutoARIMA` trend (no seasonality in the trend part), fit on a trailing window of 1344 hours (8 weeks). Full refit every 7 days; on other days the fitted ARIMA parameters are re-applied to the freshly decomposed window with `forward`. Univariate (no weather).

**D17. Serving model.** The API and refresh job serve the LSTM (probabilistic and explainable). If LightGBM is the more accurate model in the results, that is reported in the README and the model card, and the choice to serve the LSTM is stated there, not hidden.

## Engineering

**D18. Dependency pins.** This machine runs Windows Application Control, which blocked several freshly released compiled wheels (numpy 2.5, scikit-learn 1.9, lightgbm 4.7 and others). Versions were pinned to older, widely distributed releases (see `requirements.txt`); mypy runs in pure-Python mode (`MYPY_USE_MYPYC=0`, slower but identical results). This is a property of the machine, not of the project.

**D19. API serves stored forecasts.** The API reads forecasts and explanations that the refresh job computed and wrote to the artifact store, so the API image has no PyTorch and starts quickly on a scale-to-zero container. Alternative: load the model in the API. Reason: smaller image, lower memory, no cold start model load.

**D20. Test fixtures are real data.** `tests/data/` holds a slice of the real processed national table (2023-01 to 2023-07), one real SMARD chunk, one real SMARD CSV export, and one real Open-Meteo response. Model and pipeline tests use them with split dates moved inside the slice. A few tests use hand-written inputs with hand-computed answers for pure arithmetic (metrics, DM test, quarter-hour aggregation, gap filling).

**D21. Protocol note (test set exposure).** During development a smoke test printed test-period scores for the seasonal naive baselines and for LightGBM with default hyperparameters on the national series, before any hyperparameter search. No decision was taken from those numbers (the LightGBM and LSTM searches use validation only). From then on, development runs evaluated validation only; the test split is evaluated once, after all models are frozen.

**D22. Coverage threshold.** `--cov-fail-under=80` on `src/energy` and `api` (the CLI module is excluded because it is a thin wrapper). Heavy data modules (`eda.py`, `fetch.py`, parts of `build.py`) are exercised only locally, not in CI.
