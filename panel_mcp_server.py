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
"""

import asyncio
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

_calls: dict[str, int] = {}
JOBS: dict[str, dict] = {}

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


async def _call_one(client: httpx.AsyncClient, model: str, document: str, role: str):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": role},
            {"role": "user", "content": document},
        ],
    }
    try:
        r = await client.post(
            f"{AI_GATEWAY_BASE}/chat/completions",
            headers=_headers(), json=payload, timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        used = data.get("model", model)
        # Report the model that actually answered, not the one requested. OpenRouter can
        # route to a different provider, and a register that names the wrong critic is worse
        # than one that admits uncertainty.
        return model, {"ok": True, "model_reported": used, "findings": text}
    except Exception as e:  # noqa: BLE001 — one model failing must not kill the panel
        return model, {"ok": False, "model_reported": None, "findings": f"[ERROR] {e}"}


async def _run_panel(document: str, models: list[str], role: str) -> dict:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[_call_one(client, m, document, role) for m in models]
        )
    return {m: r for m, r in results}


@mcp.tool()
async def panel_models(filter: str = "") -> dict:
    """List model slugs currently available on OpenRouter.

    Slugs drift, so check here before relying on any hard-coded default. Pass `filter`
    to match a substring, e.g. 'deepseek' or 'gemini'.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{AI_GATEWAY_BASE}/models", headers=_headers(), timeout=60)
        r.raise_for_status()
        rows = r.json().get("data", [])
    slugs = sorted(m["id"] for m in rows)
    if filter:
        slugs = [s for s in slugs if filter.lower() in s.lower()]
    return {"count": len(slugs), "models": slugs}


@mcp.tool()
async def panel_review(
    document: str,
    models: list[str] | None = None,
    role: str | None = None,
) -> dict:
    """Run an adversarial panel review across models in parallel and return each verdict.

    Use this first. Wall-clock is the slowest single model, not the sum. If the panel is
    large or the document long enough to risk a client timeout, use panel_submit instead.
    """
    models = models or DEFAULT_PANEL
    role = role or DEFAULT_ROLE
    blocked = _spend_guard(len(models), document)
    if blocked:
        return {"error": blocked}
    started = time.time()
    results = await _run_panel(document, models, role)
    live = sum(1 for v in results.values() if v["ok"])
    return {
        "panel": models,
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
) -> dict:
    """Start a panel review in the background and return a job id immediately.

    Use when panel_review would exceed the client timeout. Poll panel_status with the id.
    """
    models = models or DEFAULT_PANEL
    role = role or DEFAULT_ROLE
    blocked = _spend_guard(len(models), document)
    if blocked:
        return {"error": blocked}
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "started": time.time(), "panel": models}

    async def _worker():
        try:
            res = await _run_panel(document, models, role)
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
    return {
        "server": "panel-review",
        "reachable": True,
        "ai_gateway_key_set": key_set,
        "message": "Ready." if key_set else
                   "Server is up but AI_GATEWAY_API_KEY is not set. Add it and redeploy.",
        "limits": {"max_doc_chars": MAX_DOC_CHARS, "max_models": MAX_MODELS,
                   "max_calls_per_day": MAX_CALLS_PER_DAY},
    }


if __name__ == "__main__":
    # streamable-http is required for a remote connector. The default is stdio, which
    # serves nothing over the network — the host reports the service healthy while the
    # URL returns nothing, which is a confusing failure to diagnose.
    mcp.run(transport="streamable-http")
