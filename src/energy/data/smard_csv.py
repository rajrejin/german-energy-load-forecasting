"""Second SMARD parser: the CSV export of the "Download market data" page.

Format observed on a real export: UTF-8 with BOM, ';' separated, German headers, local
(Europe/Berlin) timestamps "dd.mm.yyyy HH:MM", German numbers ("40.170,25": '.' thousands,
',' decimal) and '-' for missing values. The export endpoint used by the website is
documented by the community project bundesAPI/smard-api (unofficial).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import httpx
import pandas as pd

EXPORT_URL = "https://www.smard.de/nip-download-manager/nip/download/market-data"


def parse_german_number(x: str) -> float:
    """'40.170,25' -> 40170.25 ; '-' or '' -> NaN."""
    s = str(x).strip()
    if s in ("", "-", "nan"):
        return float("nan")
    return float(s.replace(".", "").replace(",", "."))


def parse_export(source: bytes | str | Path) -> pd.DataFrame:
    """Parse an export into a UTC-indexed frame with one column per value column.

    Column names are the German headers with unit text removed, e.g. 'Netzlast'. The index is
    the interval start in UTC. Repeated local hours on the autumn DST change are resolved by
    the (ascending) order of the rows.
    """
    raw = Path(source).read_bytes() if isinstance(source, str | Path) else source
    text = raw.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    t0 = df.columns[0]
    start = pd.to_datetime(df[t0], format="%d.%m.%Y %H:%M")
    idx = pd.DatetimeIndex(start).tz_localize("Europe/Berlin", ambiguous="infer").tz_convert("UTC")
    out = pd.DataFrame(index=idx)
    for c in df.columns[2:]:
        name = re.sub(r"\s*\[.*$", "", c).strip()
        out[name] = [parse_german_number(v) for v in df[c]]
    out.index.name = "ts_utc"
    return out


def download_export(
    filt: int,
    region: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolution: str = "hour",
    client: httpx.Client | None = None,
) -> bytes:
    """Request a CSV export (module id 5000000 + filter, verified for filter 410)."""
    body = {
        "request_form": [
            {
                "format": "CSV",
                "moduleIds": [5000000 + filt],
                "region": region,
                "timestamp_from": int(start.timestamp() * 1000),
                "timestamp_to": int(end.timestamp() * 1000) - 1,
                "type": "discrete",
                "language": "de",
                "resolution": resolution,
            }
        ]
    }
    cli = client or httpx.Client(timeout=60.0)
    r = cli.post(EXPORT_URL, json=body)
    r.raise_for_status()
    return r.content
