# Known issues and limits

1. **Bicep, Azure and both pipelines were never executed or compiled.** The Bicep CLI and `az` were not available in the build environment and a downloaded binary was not installed. See `docs/AZURE_DEPLOY.md`.
2. **Historical forecast weather is not a true day-ahead forecast.** Open-Meteo's Historical Forecast API stitches early hours of successive model runs, so the "archived forecast" regime is closer to an analysis than to a forecast issued 15 to 38 hours ahead. The previous-runs API (forecasts of previous days) would be a better realistic regime and is not used.
3. **Weather aggregation is an approximation.** 3 to 5 cities per control area with approximate populations; state shares for holidays are approximate.
4. **DST handling of lags.** Load lags are exact hour offsets in UTC. On the day after a clock change the "same hour yesterday" lag is shifted by one local hour. The breakdown "within 1 day of DST change" in `results/breakdowns.csv` quantifies the effect.
5. **Publication latency of load data is assumed to be zero** (see D9 in DECISIONS.md).
6. **Hyperparameters were searched on the national series only** and reused for the zones.
7. **Windows Application Control** on the build machine forced older pinned wheels; this affects reproducibility only on machines with the same policy.
8. **Docker** was verified after training finished: the compose stack (API, app, Azurite) built, reached healthy status and served forecasts that the refresh job wrote into Azurite. Docker Desktop must be started by hand on Windows.
9. **Live weather versus reanalysis.** The refresh job uses the live API's recent analysis for observed weather, whereas training used reanalysis; a small distribution shift that was not quantified.
10. **Briefing evaluation covers 20 of 144 test weeks** because one briefing takes about 288 s on this CPU (`results/briefing_eval.json`).
11. **Interval under-coverage on the test period** (70 to 75% for a nominal 80%): the conformal constant fit on 2023 is too narrow after the demand drift.

## Audit of 2026-10-22: problems found and fixed

Each was reproduced first and now has a regression test where it is code.

| Problem | Effect | Fix |
| --- | --- | --- |
| `scripts/wait_fetch.ps1` was blocked by the default PowerShell execution policy (`Restricted`) | the compute queue did not wait for the generation download, so the dataset was rebuilt a second time later | removed; `energy wait-fetch` waits on per-process lock files that `energy fetch` writes (`tests/unit/test_fetch_lock.py`) |
| `energy refresh --no-fetch` stopped on an `assert` | the documented option did not work | `--no-fetch` now reuses the cached SMARD load and still fetches live weather (`test_refresh_without_fetch_uses_cached_load_and_fetches_weather`) |
| `block_bootstrap_ci` raised `ValueError` for series shorter than the 7-day block | any short breakdown passed to it would crash the report | block length is capped at the series length; empty input returns NaN (`test_block_bootstrap_handles_series_shorter_than_a_block`) |
| Streamlit "MAE by hour" chart mixed both weather regimes into one line per model | misleading zigzag chart | the backtest tab filters by zone and regime; a regime switch was added |
| Streamlit used `use_container_width`, deprecated with removal announced "after 2025-12-31" | the app would break on a Streamlit upgrade | replaced with `width="stretch"`; `app/` is now linted and format-checked in CI |
| `energy fetch --slice` downloaded the hard-coded years 2023 to 2026 | the vertical slice would miss data from 2027 on | years now run to the current year |
| `infra/azure/deploy.sh` created the resource group before `--yes` | a cloud change during what was described as a preview | without `--yes` it only previews (or explains that the group does not exist); it also runs `az bicep build` first |
| HANDOFF quoted about 40 minutes for the pooled model | wrong number (the artifact records 1699 s) | now read from `results/runtime.json` |
| README said `energy fetch` takes about 30 minutes | understated (generation alone took about 2 hours here) | quick start split into load, weather and optional generation |
