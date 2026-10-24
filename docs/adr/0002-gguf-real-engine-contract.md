# GGUF goes through the engine's real contract: --gguf <file>

Status: accepted (2026-10-24, design grill)

The station launched the GGUF backend with `-ngl <computed>` and `--threads`,
and passed a repo id or snapshot directory as `--model`. The basal engine
(v2.3.5 and v2.4.1) defines none of `-ngl`/`--threads` — it exited with an
argparse error before loading weights — and takes GGUF weights via
`--gguf <file>`. The flagship "3 GB VRAM, partial offload" backend could
never start.

Decided:

1. The station resolves the actual downloaded `.gguf` file from the snapshot
   directory and passes it as `--gguf`; `--model` carries the tokenizer/repo.
2. `-ngl` and `--threads` are deleted from argument building.
3. The VRAM planner (`plan_gguf_offload`, per-layer math) stays as a
   **display-only estimate** in the console. The engine owns the real offload
   decision.

Rejected: relying on the engine's "converted from --model" auto path
(unverified behavior); bumping the engine pin as the fix (v2.4.1 has the same
contract).

Consequence: the README's offload-percentage tables describe estimates, not
flags the station sends. An engine smoke test (skipped when the engine is not
installed) guards the argument contract.
