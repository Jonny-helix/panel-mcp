# panel-review MCP server — deployment

Adversarial multi-model review via OpenRouter, callable from Claude as a custom connector.

Tested before packaging: server starts, serves `/mcp` over streamable-http, all five tools
register, and the health check reports cleanly when the API key is absent rather than crashing.

## Files

| File | Purpose |
|---|---|
| `panel_mcp_server.py` | The server. |
| `requirements.txt` | `mcp` and `httpx`. Railway installs these automatically. |
| `railway.json` | Tells Railway how to build and start. Means you don't set the start command by hand. |
| `Procfile` | Fallback for hosts that read a Procfile instead. Harmless on Railway. |

## Deploy

1. Put all four files in a GitHub repo, at the top level (not in a subfolder).
2. Railway → **New Project** → **Deploy from GitHub repo** → pick the repo.
3. Railway service → **Variables** → **New Variable**:
   `OPENROUTER_API_KEY` = your key. Do this before worrying about the first deploy failing.
4. Railway service → **Settings** → **Networking** → **Generate Domain**.
   Without this the service runs but has no public URL. Port 8000 if asked.
5. Your connector URL is `https://<the-domain-railway-gave-you>/mcp` — note the `/mcp`.

## Connect to Claude

**Customize > Connectors > + > Add custom connector**, paste the URL, Add.
Then enable it per conversation via the **+** button > Connectors.

## First thing to run

Ask Claude to run `panel_health`. It confirms the server is reachable and the key is set.
Then `panel_models` with a filter (`deepseek`, `gemini`, `gpt`) to get slugs that currently
resolve — the defaults in the file will drift and should not be trusted.

## Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PANEL_MAX_DOC_CHARS` | 400000 | Rejects oversized documents. |
| `PANEL_MAX_MODELS` | 6 | Caps models per call. |
| `PANEL_MAX_CALLS_PER_DAY` | 200 | Daily spend guard. |

## Internal and client routing

- **Internal work** — call the review tools with no `client`. Any model, any region, cheapest provider first.
- **Client material** — always pass `client` (for example `client="mantle"`). The server looks up the client's
  zone in `client_regions.json`, pins inference there with zero data retention and no prompt training, checks
  each model can serve that zone, and records the zone the gateway reports on each verdict.
  A client not in the file uses `_default` (eu). Models that cannot serve the zone are skipped with a note and
  the rest of the panel runs; nothing is ever routed outside the zone.
- **Changing a client's zone** — edit `client_regions.json` and commit, only after agreeing the change with the client.
- `panel_models(client="mantle")` lists eligible models; `panel_models(open_weights=True, detailed=True)` adds prices.

## Two things to know

**Spend exposure.** This endpoint spends your OpenRouter key and the URL is the only thing
protecting it. Anyone who learns the URL can burn credits. The caps above limit the damage but
do not prevent it. Set a spend limit in the OpenRouter dashboard as well, and treat the URL as
a secret. If it leaks, delete the Railway domain and generate a new one.

**Job state is in memory.** `panel_submit` and `panel_status` share state within one process.
A redeploy loses running jobs, and if you ever scale to more than one replica the poll can land
on the wrong instance and report "unknown". Stay on a single replica.
