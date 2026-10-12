"""Briefing generation: local LLM (Ollama, else llama-cpp-python), with verification and retry."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from energy.briefing.verify import verify

log = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5:3b-instruct"
GGUF_NAME = "qwen2.5-3b-instruct-q4_k_m.gguf"

SYSTEM = (
    "You write a short weekly briefing on forecast electricity demand for grid operators. "
    "Use ONLY the numbers, zone names and driver names that appear in the JSON facts. "
    "Copy numbers exactly as given (peaks in MW, energy in GWh, changes in percent). "
    "Do not invent numbers, do not convert units, and do not state causes that the facts do "
    "not contain. Cover every zone with exactly three short bullets: peak and hour, energy "
    "and change versus the previous week, interval width and the listed drivers. "
    "Answer with JSON only."
)


class LLMBriefing(BaseModel):
    headline: str
    zones: dict[str, list[str]]
    risks: list[str]
    attention: list[str]


class Briefing(LLMBriefing):
    fallback_template: bool = False


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = OLLAMA_MODEL, host: str | None = None):
        import ollama

        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        names = {m.model for m in self.client.list().models}
        if not any(n and n.startswith(model.split(":")[0]) for n in names):
            raise RuntimeError(f"model {model} is not pulled in Ollama")

    def complete(self, system: str, user: str) -> str:
        r = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=LLMBriefing.model_json_schema(),
            options={"temperature": 0, "seed": 0, "num_predict": 700},
        )
        return str(r.message.content)


class LlamaCppBackend:
    name = "llama-cpp-python"

    def __init__(self, gguf: str | Path, n_ctx: int = 4096, n_threads: int | None = None):
        from llama_cpp import Llama

        self.llm = Llama(
            model_path=str(gguf), n_ctx=n_ctx, n_threads=n_threads or os.cpu_count(), verbose=False
        )

    def complete(self, system: str, user: str) -> str:
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            seed=0,
            max_tokens=700,
            response_format={"type": "json_object", "schema": LLMBriefing.model_json_schema()},
        )
        return str(r["choices"][0]["message"]["content"])


def get_backend(models_dir: Path | None = None) -> Backend | None:
    """Ollama if reachable, else llama-cpp with a local GGUF, else None (template)."""
    try:
        return OllamaBackend()
    except Exception as e:
        log.info("ollama unavailable: %s", str(e)[:120])
    gguf = Path(os.environ.get("ENERGY_GGUF", "")) if os.environ.get("ENERGY_GGUF") else None
    gguf = gguf or ((models_dir or Path("models")) / GGUF_NAME)
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
        bullets = [
            f"Expected peak {f['expected_peak_mw']:.0f} MW on {f['peak_date']} at hour "
            f"{f['peak_hour_local']}.",
            f"Expected energy {f['energy_gwh']:.1f} GWh"
            + (
                f", {f['energy_change_pct']:+.1f}% versus the previous week."
                if "energy_change_pct" in f
                else "."
            ),
            f"Mean 80% interval width {f['mean_interval_width_mw']:.0f} MW."
            + (
                " Top drivers of the peak hour: "
                + ", ".join(
                    f"{d['driver']} ({d['attribution_mw']:+.0f} MW)" for d in f["peak_hour_drivers"]
                )
                + "."
                if f.get("peak_hour_drivers")
                else ""
            ),
        ]
        zones[z] = bullets
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


def user_prompt(facts: dict[str, Any], extra: str = "") -> str:
    slim = json.loads(json.dumps(facts))
    for z in slim["zones"].values():
        z.pop("peak_day_index", None)
        z.pop("peak_hour_index", None)
    return f"FACTS:\n{json.dumps(slim, indent=1)}\n{extra}\nWrite the briefing JSON now."


def generate(
    facts: dict[str, Any], backend: Backend | None, max_retries: int = 2
) -> tuple[Briefing, dict[str, Any]]:
    """Generate, verify and retry; fall back to the template. Returns the briefing and info."""
    info: dict[str, Any] = {"backend": backend.name if backend else "template", "attempts": 0}
    if backend is None:
        info.update(fallback=True, passed_first_attempt=False, reason="no language model available")
        return template_briefing(facts), info
    extra = ""
    problems: list[str] = []
    t0 = time.time()
    for attempt in range(1 + max_retries):
        info["attempts"] = attempt + 1
        raw = backend.complete(SYSTEM, user_prompt(facts, extra))
        try:
            b = LLMBriefing.model_validate_json(raw)
            problems = verify(b.model_dump(), facts)
        except (ValidationError, ValueError) as e:
            problems = [f"output is not valid against the schema: {str(e)[:100]}"]
            b = None  # type: ignore[assignment]
        info.setdefault("problems_by_attempt", []).append(problems)
        if not problems and b is not None:
            info.update(
                fallback=False,
                passed_first_attempt=(attempt == 0),
                seconds=time.time() - t0,
            )
            return Briefing(**b.model_dump()), info
        extra = (
            "Your previous answer failed verification: "
            + "; ".join(problems[:8])
            + ". Use only numbers and names from the facts, exactly as written."
        )
    info.update(fallback=True, passed_first_attempt=False, seconds=time.time() - t0)
    return template_briefing(facts), info
