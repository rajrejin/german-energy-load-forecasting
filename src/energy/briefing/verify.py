"""Numeric and naming verification of a generated briefing against the input facts."""

from __future__ import annotations

import re
from typing import Any

from energy.briefing.aggregate import FEATURE_LABELS

ZONE_TOKENS = ("50Hertz", "Amprion", "TenneT", "TransnetBW")
NUM_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
# Constants that may appear without being data: days in a week, the "top 3" of drivers.
FREE_NUMBERS = {3.0, 7.0}


def flatten_numbers(obj: Any, out: set[float] | None = None) -> set[float]:
    """All numeric leaves of the facts, plus numbers embedded in date strings."""
    out = out if out is not None else set()
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out.add(float(obj))
    elif isinstance(obj, str):
        for tok in re.findall(r"\d+", obj):
            out.add(float(tok))
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_numbers(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            flatten_numbers(v, out)
    return out


def allowed_values(facts: dict[str, Any]) -> set[float]:
    base = flatten_numbers(facts) | FREE_NUMBERS
    ext: set[float] = set()
    for v in base:
        ext.update({v, abs(v), v / 1000.0, abs(v) / 1000.0})
    return ext


def _decimals(tok: str) -> int:
    t = tok.replace(",", "")
    return len(t.split(".")[1]) if "." in t else 0


def parse_number(tok: str) -> float:
    """'12,345' and '12,345.6' use ',' as a thousands separator, '3.5' is a decimal."""
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", tok):
        return float(tok.replace(",", ""))
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", tok):
        return float(tok.replace(".", ""))
    return float(tok)


def candidates(tok: str) -> list[float]:
    """Readings of a token. '1.234' is a decimal in English and a thousands group in German."""
    vals = [parse_number(tok)]
    if re.fullmatch(r"-?\d{1,3}\.\d{3}", tok):
        vals.append(float(tok))
    return vals


def numbers_in(text: str) -> list[tuple[str, float]]:
    return [(m.group(0), parse_number(m.group(0))) for m in NUM_RE.finditer(text)]


def matches(x: float, decimals: int, allowed: set[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(x - a) <= tol for a in allowed)


def collect_strings(b: dict[str, Any]) -> list[str]:
    out = [b.get("headline", "")]
    for bullets in b.get("zones", {}).values():
        out += list(bullets)
    out += list(b.get("risks", [])) + list(b.get("attention", []))
    return out


def verify(
    briefing: dict[str, Any], facts: dict[str, Any], require_all_zones: bool = True
) -> list[str]:
    """Return a list of problems (empty when the briefing passes)."""
    problems: list[str] = []
    allowed = allowed_values(facts)
    zones_in_facts = set(facts["zones"])
    driver_labels = {
        d["driver"] for z in facts["zones"].values() for d in z.get("peak_hour_drivers", [])
    }
    for zone in briefing.get("zones", {}):
        if zone not in zones_in_facts:
            problems.append(f"zone '{zone}' does not exist in the facts")
    text_all = " ".join(collect_strings(briefing))
    for z in ZONE_TOKENS:
        if z in text_all and z not in zones_in_facts:
            problems.append(f"zone '{z}' is named but not in the facts")
    for s in collect_strings(briefing):
        for tok, _val in numbers_in(s):
            if not any(matches(v, _decimals(tok), allowed) for v in candidates(tok)):
                problems.append(f"number {tok} is not in the facts")
    all_labels = set(FEATURE_LABELS.values())
    for lab in all_labels - driver_labels:
        if lab in text_all and not any(lab in d or d in lab for d in driver_labels):
            problems.append(f"driver '{lab}' is named but is not among the provided drivers")
    # every zone in the facts must be covered
    for z in zones_in_facts if require_all_zones else []:
        if z not in briefing.get("zones", {}):
            problems.append(f"zone '{z}' is missing from the briefing")
    return sorted(set(problems))
