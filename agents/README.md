# Agent-harness kit

Wire the decision engine into any agent harness (local CLI coding agents,
server agents, MCP-capable runtimes). The engine is a **judge, not an
executor**: it returns typed probabilities; your agent's code acts on them.

## 1. MCP (rich agents)

Register the MCP server (see README §Plugging it in):

```json
"basal-station": {
  "type": "stdio",
  "command": "<install>/.venv/bin/python",
  "args": ["-m", "basal_station.mcp_server"]
}
```

Tools: `decide`, `knowledge_search`, `knowledge_ask`, `engine_status`,
`engine_start`.

## 2. Skill file (agents that read skills)

Install the bundled skill so the agent knows when and how to judge:

```bash
cp -r agents/basal-decisions <agent-skills-dir>/
```

For harnesses that auto-discover `.agents/` or `~/.agents/` directories,
symlink it there instead. Adjust the `BASAL_STATION_BIN` path below if the
install lives elsewhere.

## 3. Verify gate (shell pipelines / hooks)

Judge any produced answer before sending it:

```bash
echo '{"task": "...", "context": "...", "answer": "..."}' \
  | agents/verify_gate.py
case $? in
  0) echo "send" ;;      # accepted
  1) echo "revise" ;;    # regenerate with the printed critique
  2) echo "abstain" ;;   # retrieve more / escalate
  *) echo "engine down — fail closed" ;;
esac
```

Exit-code contract of `basal-station decide`:

| code | meaning |
|---|---|
| 0 | decided, policy not flagged |
| 3 | deferred or flagged by policy — escalate |
| other | input/engine error |

## 4. Pipeline patterns

- **Triage-first**: on every inbound item, run the triage recipe; only
  confident, low-impact, non-sensitive items proceed automatically.
- **Propose → judge → send**: any generative step's output goes through
  `verify_gate.py`; revise once on `revise`, escalate on `abstain`.
- **Capability router**: before an expensive call, ask which handler should
  take it (fast classifier / retrieval / deep reasoner / human) and enforce
  budgets in code.
- **Safety gate**: inspect inbound content for override attempts before it
  reaches the agent; block on failure.
- **Shadow mode**: log decisions + policy actions next to real outcomes for a
  week; measure with `basal-station eval` before enabling automation.

## 5. Evaluation loop

```bash
# questions.json + examples.jsonl ({state, labels} per line)
basal-station eval examples.jsonl --questions questions.json | jq
```

Reports per-question accuracy / confusion (choice), false positives and
negatives (noul), mean absolute error (score), and confidence calibration —
the "is the boundary actually good?" answer before you widen automation.
