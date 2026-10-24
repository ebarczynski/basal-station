# Adoption attaches to any healthy engine, but reports only verified facts

Status: accepted (2026-10-24, design grill)

Several clients (station, MCP server, CLI) may race for the engine port. The
station deliberately adopts whatever already answers `/health` instead of
spawning a duplicate — that part stays. But adoption stamped the *requested*
backend and model into the status, so `/api/system` reported "12B MLX engine
running" even when the port was served by a different engine (or a different
model entirely), and eligibility checks were bypassed.

Decided (owner's call, softer than verify-gate-409):

- Adopt anything healthy on the port — keep the race hedge and fast startup.
- Until the engine's `/v1/models` (or equivalent) confirms what is served,
  status reports `backend`/`model` as **unknown/adopted-unverified**. The
  request that triggered adoption is never echoed as fact.
- Once verified, status reports the engine's actual model and mode, with a
  warning when it differs from what the caller asked for.

Rejected: refusing adoption on mismatch (409) — turns a shared-port
convenience into a startup failure for legitimate multi-client setups.

Consequence: the adopted path can no longer lie; UI/API show an explicit
"adopted, unverified" state until confirmation lands.
