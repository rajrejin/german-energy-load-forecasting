# Known issues and limits

1. **Bicep, Azure and both pipelines were never executed or compiled.** The Bicep CLI and `az` were not available in the build environment and a downloaded binary was not installed. See `docs/AZURE_DEPLOY.md`.
2. **Historical forecast weather is not a true day-ahead forecast.** Open-Meteo's Historical Forecast API stitches early hours of successive model runs, so the "archived forecast" regime is closer to an analysis than to a forecast issued 15 to 38 hours ahead. The previous-runs API (forecasts of previous days) would be a better realistic regime and is not used.
3. **Weather aggregation is an approximation.** 3 to 5 cities per control area with approximate populations; state shares for holidays are approximate.
4. **DST handling of lags.** Load lags are exact hour offsets in UTC. On the day after a clock change the "same hour yesterday" lag is shifted by one local hour. The breakdown "within 1 day of DST change" in `results/breakdowns.csv` quantifies the effect.
5. **Publication latency of load data is assumed to be zero** (see D9 in DECISIONS.md).
6. **Hyperparameters were searched on the national series only** and reused for the zones.
7. **Windows Application Control** on the build machine forced older pinned wheels; this affects reproducibility only on machines with the same policy.
8. **Docker.** The Docker daemon was not running on the build machine at the start and was not started to protect the 8 GB memory budget during training; see HANDOFF.md for what was verified later.
9. **Live weather versus reanalysis.** The refresh job uses the live API's recent analysis for observed weather, whereas training used reanalysis; a small distribution shift that was not quantified.
