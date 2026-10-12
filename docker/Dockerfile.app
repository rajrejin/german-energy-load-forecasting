# syntax=docker/dockerfile:1
FROM python:3.11.9-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY docker/requirements-app.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements-app.txt

FROM python:3.11.9-slim-bookworm AS runtime
RUN useradd --create-home --uid 10001 appuser
COPY --from=build /opt/venv /opt/venv
COPY app /app
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 ENERGY_API_URL=http://api:8000
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
