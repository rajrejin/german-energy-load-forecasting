"""Briefing generation: local LLM (Ollama, else llama-cpp-python), with verification and retry.

The briefing is generated in sections (one call per zone plus one overview call) so that each
call stays short on a CPU-only laptop and a failing section can be retried on its own.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from energy.briefing.verify import verify

log = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5:3b-instruct"
GGUF_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MAX_TOKENS = 450

SYSTEM = (
    "You write parts of a short weekly briefing on forecast electricity demand for grid "
    "operators. Use ONLY the numbers, zone names and driver names that appear in the JSON "
    "facts. Copy numbers exactly as given (peaks in MW, energy in GWh, changes in percent). "
    "Do not invent numbers, do not convert units, and do not state causes that the facts do "
    "not contain. Answer with JSON only."
)

ZONE_TASK = (
    "Write exactly three short bullets for this zone: (1) expected peak with date and local "
    "hour, (2) expected energy and the change versus the previous week, (3) the mean interval "
    "width and the listed peak-hour drivers with their attributions."
)
OVERVIEW_TASK = (
    "Write a one-sentence headline for the week, one or two risks (for example the zone with "
    "the widest interval or the largest change) and one or two attention items. Mention only "
    "zones and numbers that appear in the facts."
)


class LLMBriefing(BaseModel):
    headline: str
    zones: dict[str, list[str]]
    risks: list[str]
    attention: list[str]


class ZoneBullets(BaseModel):
    bullets: list[str]


class Overview(BaseModel):
    headline: str
    risks: list[str]
    attention: list[str]


class Briefing(LLMBriefing):
    fallback_template: bool = False


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, host: str | None = None):
        import ollama

        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        names = {m.model for m in self.client.list().models}
        if not any(n and n.startswith(model.split(":")[0]) for n in names):
            raise RuntimeError(f"model {model} is not pulled in Ollama")

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema,
            options={"temperature": 0, "seed": 0, "num_predict": MAX_TOKENS},
        )
        return str(r.message.content)


class LlamaCppBackend:
    name = "llama-cpp-python"

    def __init__(self, gguf: str | Path, n_ctx: int = 4096, n_threads: int | None = None):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(gguf), n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(), verbose=False
        )

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> str:
        r: Any = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            seed=0,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object", "schema": schema},
        )
        return str(r["choices"][0]["message"]["content"])


def get_backend(models_dir: Path | None = None) -> Backend | None:
    """Ollama if reachable, else llama-cpp with a local GGUF, else None (template)."""
    try:
        return OllamaBackend()
    except Exception as e:
        log.info("ollama unavailable: %s", str(e)[:120])
    env = os.environ.get("ENERGY_GGUF")
    gguf = Path(env) if env else (models_dir or Path("models")) / GGUF_NAME
    if gguf.exists():
        try:
            return LlamaCppBackend(gguf)
        except Exception as e:
            log.info("llama-cpp unavailable: %s", str(e)[:120])
    return None


def template_briefing(facts: dict[str, Any]) -> Briefing:
    """Deterministic briefing built straight from the facts (used when verification fails)."""
    zones: dict[str, list[str]] = {}
    for z, f in facts["zones"].items():
        drivers = ""
        if f.get("peak_hour_drivers"):
            drivers = (
                " Top drivers of the peak hour: "
                + ", ".join(
                    f"{d['driver']} ({d['attribution_mw']:+.0f} MW)" for d in f["peak_hour_drivers"]
                )
                + "."
            )
        change = (
            f", {f['energy_change_pct']:+.1f}% versus the previous week."
            if "energy_change_pct" in f
            else "."
        )
        zones[z] = [
            f"Expected peak {f['expected_peak_mw']:.0f} MW on {f['peak_date']} at hour "
            f"{f['peak_hour_local']}.",
            f"Expected energy {f['energy_gwh']:.1f} GWh{change}",
            f"Mean 80% interval width {f['mean_interval_width_mw']:.0f} MW.{drivers}",
        ]
    widest = max(facts["zones"].items(), key=lambda kv: kv[1]["mean_interval_width_mw"])
    flagged = [z for z, f in facts["zones"].items() if f["quality_flags"]]
    return Briefing(
        headline=f"Weekly demand briefing {facts['week_start']} to {facts['week_end']}",
        zones=zones,
        risks=[
            f"Widest forecast uncertainty: {widest[0]} with a mean 80% interval width of "
            f"{widest[1]['mean_interval_width_mw']:.0f} MW."
        ],
        attention=[f"Data quality flags raised for {z}." for z in flagged]
        or ["No data quality flags raised."],
        fallback_template=True,
    )


def zone_prompt(zone: str, zf: dict[str, Any], week: dict[str, Any], extra: str = "") -> str:
    slim = {k: v for k, v in zf.items() if k not in ("peak_day_index", "peak_hour_index")}
    return (
        f"FACTS for zone {zone} (week {week['week_start']} to {week['week_end']}, "
        f"interval level {week['interval_level_pct']} percent):\n{json.dumps(slim, indent=1)}\n"
        f'{ZONE_TASK}\n{extra}\nReturn JSON with the key "bullets".'
    )


def overview_prompt(facts: dict[str, Any], extra: str = "") -> str:
    keys = (
        "expected_peak_mw",
        "energy_gwh",
        "energy_change_pct",
        "mean_interval_width_mw",
        "quality_flags",
    )
    slim = {
        "week_start": facts["week_start"],
        "week_end": facts["week_end"],
        "zones": {z: {k: f[k] for k in keys if k in f} for z, f in facts["zones"].items()},
    }
    return (
        f"FACTS:\n{json.dumps(slim, indent=1)}\n{OVERVIEW_TASK}\n{extra}\n"
        'Return JSON with the keys "headline", "risks", "attention".'
    )


def _try(
    backend: Backend,
    prompt_fn: Callable[[str], str],
    schema: dict[str, Any],
    model: type[BaseModel],
    check: Callable[[Any], list[str]],
    max_retries: int,
) -> tuple[Any, dict[str, Any]]:
    """Generate one section, verify it, retry with an explicit instruction. None = failed."""
    extra = ""
    info: dict[str, Any] = {"attempts": 0, "problems": []}
    for attempt in range(1 + max_retries):
        info["attempts"] = attempt + 1
        raw = backend.complete(SYSTEM, prompt_fn(extra), schema)
        obj: Any
        try:
            obj = model.model_validate_json(raw)
            problems = check(obj)
        except (ValidationError, ValueError) as e:
            obj, problems = None, [f"output is not valid against the schema: {str(e)[:100]}"]
        info["problems"].append(problems)
        if obj is not None and not problems:
            info["passed_first_attempt"] = attempt == 0
            return obj, info
        extra = (
            "Your previous answer failed verification: "
            + "; ".join(problems[:6])
            + ". Use only numbers and names from the facts, exactly as written."
        )
    info["passed_first_attempt"] = False
    return None, info


def generate(
    facts: dict[str, Any], backend: Backend | None, max_retries: int = 2
) -> tuple[Briefing, dict[str, Any]]:
    """Per-zone calls plus one overview call; each section is verified and retried, and a
    section that still fails is replaced by its deterministic template (flagged)."""
    tmpl = template_briefing(facts)
    info: dict[str, Any] = {"backend": backend.name if backend else "template", "attempts": 0}
    if backend is None:
        info.update(fallback=True, passed_first_attempt=False, reason="no language model available")
        return tmpl, info
    t0 = time.time()
    zones: dict[str, list[str]] = {}
    sections: dict[str, dict[str, Any]] = {}
    fallback_sections: list[str] = []
    for z, zf in facts["zones"].items():
        sub = {**facts, "zones": {z: zf}}

        def check_zone(o: ZoneBullets, z: str = z, sub: dict[str, Any] = sub) -> list[str]:
            return verify(
                {"headline": "", "zones": {z: o.bullets}, "risks": [], "attention": []}, sub
            )

        def zprompt(extra: str, z: str = z, zf: dict[str, Any] = zf) -> str:
            return zone_prompt(z, zf, facts, extra)

        obj, si = _try(
            backend,
            zprompt,
            ZoneBullets.model_json_schema(),
            ZoneBullets,
            check_zone,
            max_retries,
        )
        sections[z] = si
        if obj is None:
            zones[z] = tmpl.zones[z]
            fallback_sections.append(z)
        else:
            zones[z] = obj.bullets

    def check_overview(o: Overview) -> list[str]:
        return verify(
            {"headline": o.headline, "zones": {}, "risks": o.risks, "attention": o.attention},
            facts,
            require_all_zones=False,
        )

    def ov_prompt(extra: str) -> str:
        return overview_prompt(facts, extra)

    ov, oi = _try(
        backend,
        ov_prompt,
        Overview.model_json_schema(),
        Overview,
        check_overview,
        max_retries,
    )
    sections["overview"] = oi
    if ov is None:
        headline, risks, attention = tmpl.headline, tmpl.risks, tmpl.attention
        fallback_sections.append("overview")
    else:
        headline, risks, attention = ov.headline, ov.risks, ov.attention
    info["sections"] = sections
    info["attempts"] = sum(s["attempts"] for s in sections.values())
    info["fallback_sections"] = fallback_sections
    info["fallback"] = bool(fallback_sections)
    info["passed_first_attempt"] = all(s["passed_first_attempt"] for s in sections.values())
    info["seconds"] = time.time() - t0
    b = Briefing(
        headline=headline,
        zones=zones,
        risks=risks,
        attention=attention,
        fallback_template=bool(fallback_sections),
    )
    info["final_problems"] = verify(b.model_dump(), facts)
    return b, info
