# Grounding calls a real, verifiable KP tool — or fails visibly

Status: accepted (2026-10-24, design grill)

The station grounded decisions by calling a knowledge-palace MCP tool named
`context`. The real knowledge-palace server exposes only `search`, `similar`,
and `ask`, and answers unknown tools with the plain text `Unknown tool:
context` — which the station's hit normalizer accepted as a retrieval result.
Every grounded decision state was silently poisoned with a fake "Reference
knowledge" block while the response reported `used: true`.

Decided:

1. knowledge-palace gains a `context` tool returning **structured JSON hits**
   (text, source/title, score) for decision grounding.
2. The station checks the server's tool list at connect time. Grounding runs
   only when the required tool is actually listed; otherwise it fails visibly
   (`used: false` + error in the knowledge block). The decision itself still
   proceeds ungrounded — grounding failure must never fabricate knowledge.

Rejected: mapping grounding onto `search` (hits stay human-formatted text,
no schema to verify); hard-failing the whole decision when grounding is
unavailable (grounding is an enhancement, not a precondition).

Consequence: a regression test must feed the station a KP stub without
`context` and assert `used: false` — never a poisoned grounded state.
