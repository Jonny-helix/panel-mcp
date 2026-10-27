"""
Panel-review MCP server — adversarial multi-model review via OpenRouter.

Deploy to a PUBLIC HTTPS host (Railway). The connector URL is https://<domain>/mcp

Set OPENROUTER_API_KEY in the host's environment variables.
Never hard-code the key here and never paste it into a chat.

Tools exposed:
  panel_models  - list currently available model slugs (slugs drift; check before relying on them)
  panel_review  - run the panel in parallel, one shot. Use this first.
  panel_submit  - start a panel in the background, returns a job id
  panel_status  - poll a job id until done. Use when panel_review risks a client timeout.

Two routing modes:
  internal (no `client`) - any model, any region, cheapest provider first.
  client (`client` set)  - the client's zone from client_regions.json, zero data retention
                           and no prompt training. Refuses to run rather than route elsewhere.
"""

import asyncio
import json
import os
import time
import uuid

import httpx
from mcp.server.fastmcp import FastMCP

AI_GATEWAY_BASE = "https://ai-gateway.vercel.sh/v1"

# Read lazily, not at import. Reading at import means a missing variable kills the
# process on startup and the host crash-loops with an error that points at the wrong thing.
def _api_key() -> str:
    key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "AI_GATEWAY_API_KEY is not set on this server. Add it in the host's "
            "environment variables and redeploy."
        )
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


DEFAULT_ROLE = (
    "You are an adversarial critic reviewing the document below. Surface flaws only; "
    "do not praise and do not summarise what the document says. Classify every finding "
    "by severity: Blocker, Major, or Minor. Be specific about what is wrong and what "
    "would fix it. Return findings as a list, one per line, formatted: [SEVERITY] finding."
)

# Slugs drift. These are a starting point only — call panel_models to see what currently
# resolves, and pass an explicit `models` list rather than trusting this default.
DEFAULT_PANEL = [
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k3",
    "xiaomi/mimo-v2.6-pro",
    "openai/gpt-6-astra",
]

# Spend guards. This endpoint spends a real API key and its URL is the only thing
# protecting it, so cap the blast radius rather than relying on the URL staying secret.
MAX_DOC_CHARS = int(os.environ.get("PANEL_MAX_DOC_CHARS", "400000"))
MAX_MODELS = int(os.environ.get("PANEL_MAX_MODELS", "6"))
MAX_CALLS_PER_DAY = int(os.environ.get("PANEL_MAX_CALLS_PER_DAY", "200"))
# Output cap per model call (includes any reasoning tokens). Long deliverables and
# reasoning models need far more than the old 8,000-token ceiling.
MAX_OUTPUT_TOKENS = int(os.environ.get("PANEL_MAX_OUTPUT_TOKENS", "32000"))
# Per-model time limits (seconds): overall cap, and the longest silence allowed while streaming.
CALL_TIMEOUT_S = float(os.environ.get("PANEL_CALL_TIMEOUT_S", "1500"))
READ_IDLE_S = float(os.environ.get("PANEL_READ_IDLE_S", "600"))

_calls: dict[str, int] = {}
JOBS: dict[str, dict] = {}

# ---- Client regions -------------------------------------------------------------------
# The register is a file in the repo so every change to a client's zone is a commit you
# made on purpose. Re-read on every call, so an edit takes effect on the next redeploy
# without code changes.
REGIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_regions.json")
VALID_ZONES = {"us", "eu", "au", "jp"}


DEFAULT_CLIENT_ZONE = "eu"  # UK/EU business: used if the file has no _default line


def _load_regions_file() -> dict:
    with open(REGIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _client_regions() -> dict[str, str]:
    raw = _load_regions_file()
    return {k.strip().lower(): v.strip().lower()
            for k, v in raw.items() if not k.startswith("_")}


def _default_zone() -> str:
    return str(_load_regions_file().get("_default", DEFAULT_CLIENT_ZONE)).strip().lower()


def _routing_for(client: str | None) -> tuple[dict | None, str | None, str | None]:
    """Return (gateway_options, zone, error). No client = internal, cost-first routing.
    A client not in the register gets the default zone (eu) rather than being refused."""
    if not client:
        return {"sort": "cost"}, None, None
    zone = _client_regions().get(client.strip().lower()) or _default_zone()
    if zone not in VALID_ZONES:
        return None, None, f"client '{client}' has invalid zone '{zone}' (allowed: {sorted(VALID_ZONES)})"
    return {
        "inferenceRegion": {"scope": "zone", "geoRegion": zone},
        "zeroDataRetention": True,
        "disallowPromptTraining": True,
    }, zone, None


# ---- Model catalogue (for pre-flight checks) ------------------------------------------
_catalogue: dict = {"at": 0.0, "rows": {}}
CATALOGUE_TTL_S = 600


async def _models_catalogue() -> dict[str, dict]:
    if time.time() - _catalogue["at"] > CATALOGUE_TTL_S or not _catalogue["rows"]:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{AI_GATEWAY_BASE}/models", headers=_headers(), timeout=60)
            r.raise_for_status()
            _catalogue["rows"] = {m["id"]: m for m in r.json().get("data", [])}
            _catalogue["at"] = time.time()
    return _catalogue["rows"]


def _model_fits_zone(row: dict | None, zone: str) -> str | None:
    """None if the model can serve this client zone with zero data retention, else why not."""
    if row is None:
        return "not in the gateway's model list"
    if zone not in (row.get("regions") or []):
        return f"cannot be pinned to '{zone}' (zones: {row.get('regions') or 'none'})"
    if row.get("zdr") == "none":
        return "has no zero-data-retention provider"
    return None


async def _preflight(models: list[str], zone: str | None) -> tuple[list[str], dict, str | None]:
    """Split the requested models into those that can serve the zone and those skipped.
    Only stops if no requested model can serve the zone."""
    if zone is None:
        return models, {}, None
    rows = await _models_catalogue()
    skipped = {m: why for m in models if (why := _model_fits_zone(rows.get(m), zone))}
    usable = [m for m in models if m not in skipped]
    if usable:
        return usable, skipped, None
    return [], skipped, (f"none of the requested models can serve zone '{zone}'. "
                         f"Call panel_models with client set to see eligible models.")


def _skip_note(skipped: dict) -> str | None:
    if not skipped:
        return None
    return ("Skipped for this client's zone (the rest of the panel ran): "
            + "; ".join(f"{m} {why}" for m, why in skipped.items()))


def _find_zone(obj) -> str | None:
    """Find gateway routing metadata (inferenceEndpoint.geoRegion) anywhere in a chunk."""
    if isinstance(obj, dict):
        ep = obj.get("inferenceEndpoint")
        if isinstance(ep, dict) and ep.get("geoRegion"):
            return ep["geoRegion"]
        for v in obj.values():
            if (z := _find_zone(v)):
                return z
    elif isinstance(obj, list):
        for v in obj:
            if (z := _find_zone(v)):
                return z
    return None

mcp = FastMCP(
    name="panel-review",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "8000")),  # Railway injects PORT
)


def _spend_guard(n_models: int, doc: str) -> str | None:
    if len(doc) > MAX_DOC_CHARS:
        return f"document is {len(doc)} chars, over the {MAX_DOC_CHARS} limit"
    if n_models > MAX_MODELS:
        return f"{n_models} models requested, over the {MAX_MODELS} limit"
    day = time.strftime("%Y-%m-%d")
    used = _calls.get(day, 0)
    if used + n_models > MAX_CALLS_PER_DAY:
        return f"daily call cap reached ({used}/{MAX_CALLS_PER_DAY})"
    _calls[day] = used + n_models
    return None


async def _call_one(client: httpx.AsyncClient, model: str, document: str, role: str,
                    gateway_opts: dict | None = None, zone: str | None = None):
    """Stream the response. Slow reasoning models were cut off at about 340 s when the
    reply came back in one piece (the gateway drops a silent connection); a stream keeps
    the connection busy, so only the overall cap applies."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": role},
            {"role": "user", "content": document},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }
    if gateway_opts:
        payload["providerOptions"] = {"gateway": gateway_opts}
    parts: list[str] = []
    used = model
    ran_in: str | None = None
    async def _stream():
        nonlocal used, ran_in
        timeout = httpx.Timeout(READ_IDLE_S, connect=30.0)
        async with client.stream(
            "POST", f"{AI_GATEWAY_BASE}/chat/completions",
            headers=_headers(), json=payload, timeout=timeout,
        ) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"HTTP {r.status_code}: {body}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk)
                except ValueError:
                    continue
                used = data.get("model", used) or used
                ran_in = _find_zone(data) or ran_in
                for ch in data.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)

    try:
        await asyncio.wait_for(_stream(), timeout=CALL_TIMEOUT_S)
        text = "".join(parts)
        if not text.strip():
            raise RuntimeError("empty response from the model")
        if zone and ran_in and ran_in != zone:
            # The gateway is meant to fail rather than do this; withhold the output if it does.
            raise RuntimeError(f"ran in zone '{ran_in}', not the client's zone '{zone}'; output withheld")
        # Report the model that actually answered, not the one requested.
        return model, {"ok": True, "model_reported": used, "findings": text,
                       "zone_requested": zone, "zone_reported": ran_in}
    except Exception as e:  # noqa: BLE001 — one model failing must not kill the panel
        partial = "".join(parts)
        if zone and ran_in and ran_in != zone:
            partial = ""  # never return text produced outside the client's zone
        note = f"[ERROR] {type(e).__name__}: {e}" + (f"\n\n[PARTIAL RESPONSE BEFORE THE ERROR]\n{partial}" if partial else "")
        return model, {"ok": False, "model_reported": None, "findings": note,
                       "zone_requested": zone, "zone_reported": ran_in}


async def _run_panel(document: str, models: list[str], role: str,
                     gateway_opts: dict | None = None, zone: str | None = None) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_call_one(client, m, document, role, gateway_opts, zone) for m in models]
        )
    return {m: r for m, r in results}


def _routing_summary(client: str | None, zone: str | None) -> dict:
    if zone:
        return {"mode": "client", "client": client, "zone": zone,
                "zero_data_retention": True, "no_prompt_training": True,
                "note": "zone_reported on each result is the gateway's own record of where it ran; "
                        "null means the gateway did not report it on the stream."}
    return {"mode": "internal", "zone": "any", "provider_order": "cheapest first"}


@mcp.tool()
async def panel_models(filter: str = "", client: str = "", open_weights: bool = False,
                       detailed: bool = False) -> dict:
    """List model slugs currently available on the AI Gateway.

    Slugs drift, so check here before relying on any hard-coded default. Pass `filter`
    to match a substring, e.g. 'deepseek' or 'gemini'. Pass `client` to list only models
    that can serve that client's agreed zone with zero data retention. Pass
    `open_weights=True` for open-weight models only. Pass `detailed=True` for price
    (USD per million tokens), zones and retention per model, to pick a cheap panel.
    """
    _, zone, err = _routing_for(client or None)
    if err:
        return {"error": err}
    rows = await _models_catalogue()
    picked = []
    for mid, m in rows.items():
        if m.get("type") != "language" or mid.endswith("-batch"):
            continue
        if filter and filter.lower() not in mid.lower():
            continue
        if open_weights and "open-weights" not in (m.get("tags") or []):
            continue
        if zone and _model_fits_zone(m, zone):
            continue
        picked.append(m)
    picked.sort(key=lambda m: m["id"])
    out = {"count": len(picked), "zone": zone or "any", "models": [m["id"] for m in picked]}
    if detailed:
        def per_m(x):
            try:
                return round(float(x) * 1_000_000, 3)
            except (TypeError, ValueError):
                return None
        out["details"] = [{
            "id": m["id"],
            "input_usd_per_m": per_m((m.get("pricing") or {}).get("input")),
            "output_usd_per_m": per_m((m.get("pricing") or {}).get("output")),
            "zones": m.get("regions") or [],
            "zdr": m.get("zdr"),
            "open_weights": "open-weights" in (m.get("tags") or []),
        } for m in picked]
    return out


@mcp.tool()
async def panel_review(
    document: str,
    models: list[str] | None = None,
    role: str | None = None,
    client: str | None = None,
) -> dict:
    """Run an adversarial panel review across models in parallel and return each verdict.

    Use this first. Wall-clock is the slowest single model, not the sum. If the panel is
    large or the document long enough to risk a client timeout, use panel_submit instead.
    Leave `client` empty for internal work (any region, cheapest provider). Set `client`
    for ANY client material: it pins the client's agreed zone with zero data retention.
    """
    models = models or DEFAULT_PANEL
    role = role or DEFAULT_ROLE
    gateway_opts, zone, err = _routing_for(client)
    if err:
        return {"error": err}
    models, skipped, err = await _preflight(models, zone)
    if err:
        return {"error": err, "skipped": skipped}
    blocked = _spend_guard(len(models), document)
    if blocked:
        return {"error": blocked}
    started = time.time()
    results = await _run_panel(document, models, role, gateway_opts, zone)
    live = sum(1 for v in results.values() if v["ok"])
    return {
        "panel": models,
        "routing": _routing_summary(client, zone),
        "skipped": _skip_note(skipped),
        "live_critics": live,
        "elapsed_seconds": round(time.time() - started, 1),
        "warning": None if live >= 2 else
                   "FEWER THAN 2 LIVE CRITICS — this is not a panel. Do not present it as one.",
        "results": results,
    }


@mcp.tool()
async def panel_submit(
    document: str,
    models: list[str] | None = None,
    role: str | None = None,
    client: str | None = None,
) -> dict:
    """Start a panel review in the background and return a job id immediately.

    Use when panel_review would exceed the client timeout. Poll panel_status with the id.
    `client` works as in panel_review: empty for internal work, set for client material.
    """
    models = models or DEFAULT_PANEL
    role = role or DEFAULT_ROLE
    gateway_opts, zone, err = _routing_for(client)
    if err:
        return {"error": err}
    models, skipped, err = await _preflight(models, zone)
    if err:
        return {"error": err, "skipped": skipped}
    blocked = _spend_guard(len(models), document)
    if blocked:
        return {"error": blocked}
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "started": time.time(), "panel": models,
                    "routing": _routing_summary(client, zone), "skipped": _skip_note(skipped)}

    async def _worker():
        try:
            res = await _run_panel(document, models, role, gateway_opts, zone)
            JOBS[job_id].update(status="done", results=res,
                               elapsed_seconds=round(time.time() - JOBS[job_id]["started"], 1))
        except Exception as e:  # noqa: BLE001
            JOBS[job_id].update(status="failed", error=str(e))

    asyncio.create_task(_worker())
    return {"job_id": job_id, "status": "running", "panel": models,
            "note": "Poll panel_status with this job_id."}


@mcp.tool()
async def panel_status(job_id: str) -> dict:
    """Check a panel_submit job. Returns status running, done, or failed."""
    job = JOBS.get(job_id)
    if not job:
        return {"error": f"unknown job_id {job_id}",
                "note": "Job state is in-memory. A redeploy or a second replica loses it."}
    return job


@mcp.tool()
async def panel_health() -> dict:
    """Confirm the server is reachable and configured. Call this first after deploying."""
    key_set = bool(os.environ.get("AI_GATEWAY_API_KEY", "").strip())
    try:
        regions = {**_client_regions(), "(any other client)": _default_zone()}
    except Exception as e:  # noqa: BLE001 — report, don't crash the health check
        regions = {"error": f"client_regions.json unreadable: {e}"}
    return {
        "client_regions": regions,
        "server": "panel-review",
        "reachable": True,
        "ai_gateway_key_set": key_set,
        "message": "Ready." if key_set else
                   "Server is up but AI_GATEWAY_API_KEY is not set. Add it and redeploy.",
        "limits": {"max_doc_chars": MAX_DOC_CHARS, "max_models": MAX_MODELS,
                   "max_calls_per_day": MAX_CALLS_PER_DAY,
                   "max_output_tokens": MAX_OUTPUT_TOKENS},
    }


if __name__ == "__main__":
    # streamable-http is required for a remote connector. The default is stdio, which
    # serves nothing over the network — the host reports the service healthy while the
    # URL returns nothing, which is a confusing failure to diagnose.
    mcp.run(transport="streamable-http")
