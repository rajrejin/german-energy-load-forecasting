# Thin wrappers around the `energy` CLI.
PY ?= python
.PHONY: install fetch dataset quality eda train backtest explain report serve app test lint typecheck gate

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .
fetch:
	energy fetch
dataset:
	energy build-dataset
quality:
	energy quality
eda:
	energy eda
train:
	energy train
backtest:
	energy backtest
explain:
	energy explain
report:
	energy report
serve:
	energy serve-api
app:
	streamlit run app/streamlit_app.py
lint:
	ruff check src api tests && ruff format --check src api tests
typecheck:
	mypy
test:
	pytest
gate: lint typecheck test
