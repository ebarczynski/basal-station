# Basal Station

The local control plane around basal typed-decision models: it picks a backend,
runs the engine, grounds decisions in knowledge-palace, and turns model signals
into audited actions.

## Language

### Decisions

**State**:
The input the model judges — a message, document, transcript, or media clip.
Treated as data, never as instructions.
_Avoid_: prompt, input, context

**Question**:
One typed ask over the state: `choice`, `noul` (yes/no), `score`, `multi`,
`act`, `tool`.
_Avoid_: task, query

**Decision**:
One request: a state plus questions, answered in a single forward pass.
_Avoid_: inference, completion

**Answer**:
Per-question result: the chosen option, calibrated probabilities, confidence,
and the `deferred` flag.
_Avoid_: output, response

**Deferral**:
The model abstaining because certified probability (`max_error`) was not
reached. A deferral escalates; it never guesses.
_Avoid_: rejection, refusal

### Policy

**Policy**:
Code that turns answers into exactly one action plus the reasons for it.
The model's output is a signal, never a permission token.
_Avoid_: rules, config

**Recipe**:
A packaged question set + policy for a recurring role (triage, router, gate,
judge, ground).
_Avoid_: template, preset

**Flagged**:
The policy took a non-automated path (deferred, failed check, or fallback).
Derived from which branch fired — never from the action's name.

### Infrastructure

**Engine**:
The `basal-serve` process serving `/v1/systemone` on the engine port.

**Backend**:
A basal-serve mode + weight-format pairing for a machine class (`mlx`,
`gguf-cuda`, `gguf-metal`, `fp8-vllm`, `eager`, `mps`, `gguf-cpu`).
_Avoid_: mode (mode is the engine-side value only)

**Adoption**:
Attaching the station to an engine already listening on the port. Adopted
engines report only verified facts until `/v1/models` confirms what serves.

**Grounding**:
Retrieving knowledge-palace passages and appending them to the state before a
decision. Grounding that cannot be verified structurally must fail visibly,
never silently.

**Knowledge-palace (KP)**:
The user's personal knowledge base, reached as an MCP stdio server exposing
`context`, `search`, `ask`, `similar`.

**Catalog**:
The embedded list of basal model family entries with per-format HF repos.

**Offload plan**:
A display-only VRAM estimate for GGUF partial offload. The engine owns the
real offload decision.
_Avoid_: ngl plan (the station never passes `-ngl`)
