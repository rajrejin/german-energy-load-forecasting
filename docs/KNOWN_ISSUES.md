# Known issues and limits

1. **Bicep, Azure and both pipelines were never executed or compiled.** I did not have the Bicep CLI or `az` installed. See `docs/AZURE_DEPLOY.md`.
2. **Historical forecast weather is not a true day-ahead forecast.** Open-Meteo's Historical Forecast API stitches early hours of successive model runs, so the "archived forecast" regime is closer to an analysis than to a forecast issued 15 to 38 hours ahead. The previous-runs API (forecasts of previous days) would be a better realistic regime and is not used.
3. **Weather aggregation is an approximation.** 3 to 5 cities per control area with approximate populations; state shares for holidays are approximate.
4. **DST handling of lags.** Load lags are exact hour offsets in UTC. On the day after a clock change the "same hour yesterday" lag is shifted by one local hour. The breakdown "within 1 day of DST change" in `results/breakdowns.csv` quantifies the effect.
5. **Publication latency of load data is assumed to be zero** (see D9 in DECISIONS.md).
6. **Hyperparameters were searched on the national series only** and reused for the zones.
7. **Windows Application Control** on my laptop forced older pinned wheels; this affects reproducibility only on machines with the same policy.
8. **Docker** was verified after training finished: the compose stack (API, app, Azurite) built, reached healthy status and served forecasts that the refresh job wrote into Azurite. Docker Desktop must be started by hand on Windows.
9. **Live weather versus reanalysis.** The refresh job uses the live API's recent analysis for observed weather, whereas training used reanalysis; a small distribution shift that was not quantified.
10. **Briefing evaluation covers 20 of 144 test weeks** because one briefing takes about 288 s on this CPU (`results/briefing_eval.json`).
11. **Interval under-coverage on the test period** (70 to 75% for a nominal 80%): the conformal constant fit on 2023 is too narrow after the demand drift.
