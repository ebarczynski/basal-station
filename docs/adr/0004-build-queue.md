# Post-grill build queue — items 1–7 DONE, item 8 GATED

Status: **1–7 implemented and pushed to `grill`** (2026-10-24, tests green
at each commit). Item 8 touches the separate knowledge-palace repo and
stays gated on its own branch + confirmation.

Settled decisions live in ADR-0001 (grounding), ADR-0002 (GGUF contract),
ADR-0003 (adoption). Work happened on the `grill` branch only; `main` is
frozen until the owner merges.

Priority order:

1. Station grounding guard: refuse grounding with `used: false` when the KP
   tool list lacks `context`; regression test with a KP stub without the tool
   must assert no poisoning (ADR-0001, station side).
2. GGUF contract: resolve the downloaded `.gguf` file → `--gguf <file>`;
   delete `-ngl`/`--threads`; fix the dead `gguf_path` plumbing; engine smoke
   test, skipped when the engine is absent (ADR-0002).
3. `orders` → engine startup `extra_args`; correct the schema comment and
   README (the request-body field is ignored by the engine — measured).
4. Adoption: report `adopted/unverified` until `/v1/models` confirms; never
   echo the request as fact (ADR-0003).
5. `flagged` derived from policy branch booleans, not action strings;
   regression test for the confident-route case.
6. `/api/decide` auto-starts the engine once and retries (mirror MCP server).
7. Small batch: enrich 6000-char budget clamp; `mcp>=2.0` floor; eval noul
   fallback key fix; exact GGUF quant filename match; README launchd example
   gains `--host 127.0.0.1`.
8. knowledge-palace repo (separate repo, own branch, own confirmation):
   structured `context` tool.

Deferred (tripwire): `fast`/CUDA-graph backend mode and `--mode mlx` +
`-MLX-8bit` pairing — revisit when the engine's model loader is confirmed
against an installed engine (F16).
