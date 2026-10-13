# Model card: day-ahead demand forecasters

All numbers come from `results/` (file names in brackets).

## Intended use

Day-ahead hourly grid load forecasts for the German control areas 50Hertz, Amprion, TenneT, TransnetBW and the national total, as a portfolio and learning project. Not validated for operational dispatch, trading or market nominations.

## Models

| Model | Role | Notes |
| --- | --- | --- |
| Seasonal naive (week, day) | baselines | previous-day variant falls back to 48 h when 24 h is not yet observed at issue time |
| MSTL-ARIMA | classical baseline | statsforecast MSTL with periods 24 and 168, AutoARIMA trend, 8 week window, weekly refit |
| LightGBM | most accurate | one model per zone, relative targets, quantile models for 10% and 90% |
| Quantile LSTM | served and explained | 1 or 2 layers (selected: 2 layers, 96 hidden units, dropout 0.1), embeddings for hour, weekday, holiday; selected by validation pinball loss in a 20-trial random search on the national series (`results/hpo/`) |
| Pooled LSTM | comparison | one network for four zones with a zone embedding |

Serving the LSTM rather than LightGBM is a choice made for explainability (SHAP GradientExplainer on a neural network was a project requirement), not for accuracy. LightGBM has lower error in every zone (`results/ladder.csv`): national MAE 1306 MW versus 1510 MW, difference 204 MW with CI 151 to 248 MW. The LSTM's pinball loss at the median is also higher (755 versus 653 MW). The justification for keeping the LSTM is therefore its role as an explained model with a non-crossing quantile head, and it is clearly better than the naive and MSTL baselines (CI of the MAE gap to MSTL-ARIMA: 401 to 705 MW). If you only need accuracy, use the LightGBM artifacts.

## Data

SMARD grid load 2015-01-01 to 2026-10-12 (hourly), Open-Meteo reanalysis and archived forecasts, population-weighted over 3 to 5 cities per zone. Train 2015 to 2022, validation 2023, test 2024-01-01 to 2026-10-11 (1014 forecast days).

## Evaluation (test period)

National total, MAE in MW (95% block bootstrap CI): LSTM 1510 (1450 to 1565); LightGBM 1306 (1255 to 1358); MSTL-ARIMA 2067; seasonal naive week 2696. MAPE 2.87% (LSTM) and 2.49% (LightGBM). LSTM seed spread over 5 seeds: MAE 1507 plus or minus 55 MW (`results/seed_spread.csv`). Realistic weather regime (archived forecasts) changes the LSTM MAE by 6 MW (1504 versus 1510). Zones and breakdowns are in the README and the CSV files.

Known weaknesses of the LSTM (national, `results/breakdowns.csv`): hottest 5% of days (MAE 1617 MW versus 1216 for LightGBM); positive bias of 539 MW (over-forecasting) on the test period; error rises in 2026 (1686 MW versus 1426 in 2024).

Interval calibration: 80% intervals are widened or narrowed with one constant fit on validation. Test coverage is 71.9% (LSTM) and 69.5% (LightGBM) nationally, so the intervals are too narrow after the demand drift between 2023 and the test years.

## Explanations

SHAP GradientExplainer attributions, aggregated per feature (`results/explain_sanity.json`): permutation agreement (Spearman 0.90), both direction checks and background stability hold; additivity gap is 17.2% of the output deviation and fails a 10% threshold. Attributions explain the model, not the physical causes of demand.

## Limits and risks

* Weather per zone is a rough population-weighted mean; real operators use finer data.
* The model assumes zero publication latency for the latest observed load.
* Trained on data containing the 2020 and 2022 shifts; a new regime change (heat pumps, rooftop solar growth, new large loads) will degrade accuracy until retrained.
* Not evaluated under outages, extreme storms or holidays outside the test window beyond what the test period contains.

## Artifacts

Each model directory under `artifacts/models/<zone>/<name>/` has a `manifest.json` with config hash, data hash, seed, library versions, training period, git commit, validation metrics and file hashes; the refresh job refuses missing, incomplete or inconsistent manifests.
