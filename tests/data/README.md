# Test fixtures

`de_sample_2023.parquet` is a slice (2023-01-01 to 2023-07-31, hourly UTC) of the real national
(`DE`) table produced by `energy build-dataset`: SMARD grid load (Bundesnetzagentur, CC BY 4.0,
"Bundesnetzagentur | SMARD.de"), Open-Meteo reanalysis and archived forecast weather
(Zippenfenig 2023; ERA5, Hersbach et al. 2023, "Generated using Copernicus Climate Change Service
information"), and calendar features. It is real data, not synthetic. Tests that train or evaluate
models use it with split dates moved inside the sample (see `tests/conftest.py`).
