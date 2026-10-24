#!/usr/bin/env python3
"""Verify-gate: judge a produced answer before it is sent.

Drop-in step for agent pipelines and hooks. Reads JSON on stdin:

    {"task": "what the user asked / what the artifact must satisfy",
     "context": "evidence the answer was built from (optional)",
     "answer": "the produced text to judge"}

Exits with a machine-readable code:
    0  accept     — send it
    1  revise     — regenerate with the printed critique
    2  abstain    — evidence insufficient; retrieve more or escalate
    4  error      — engine unreachable / bad input (fail closed: do NOT send)

Usage in a pipeline:
    propose-answer | verify_gate.py || handle-rejection
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

STATION = os.environ.get(
    "BASAL_STATION_BIN",
    os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "basal-station"),
)

QUESTIONS = {
    "acceptable": {
        "type": "noul",
        "instructions": "Is this answer safe and correct to send to the user as-is?",
    },
    "groundedness": {
        "type": "score",
        "instructions": "Is every claim supported by the provided context?",
        "criteria": {
            "0.0": "contradicts or invents facts",
            "0.5": "partially supported, adds unsourced detail",
            "1.0": "fully supported by the context",
        },
    },
    "completeness": {
        "type": "score",
        "instructions": "Does it answer everything the task requires?",
        "criteria": {"0.0": "misses the task", "0.5": "partially answers", "1.0": "fully answers"},
    },
}

POLICY = {
    "noul_checks": {"acceptable": {"expect": True, "min": 0.7}},
    "score_checks": {"groundedness": {"min": 0.5}, "completeness": {"min": 0.5}},
    "check_action": "revise",
    "defer_action": "abstain",
    "min_confidence": 0.55,
    "fallback": "abstain",
}


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"bad input JSON: {exc}"}))
        return 4

    task = (data.get("task") or "").strip()
    answer = (data.get("answer") or "").strip()
    if not task or not answer:
        print(json.dumps({"error": "task and answer are required"}))
        return 4

    state = (
        f"User task / question:\n{task}\n\n"
        f"Retrieved context the answer was built from:\n{data.get('context') or '(none provided)'}\n\n"
        f"Answer to judge:\n{answer}"
    )
    payload = json.dumps({"state": state, "questions": QUESTIONS, "policy": POLICY})

    try:
        proc = subprocess.run(
            [STATION, "decide", "-"], input=payload, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"error": f"engine call failed: {exc}"}))
        return 4
    if proc.returncode not in (0, 3):
        print(json.dumps({"error": proc.stderr.strip()[:300] or "engine error"}))
        return 4

    result = json.loads(proc.stdout)
    answers = result.get("answers", {})
    policy = result.get("policy", {})
    action = policy.get("action", "abstain")

    critique = []
    for name, a in answers.items():
        if a.get("type") == "score":
            value = a.get("score", a.get("choice"))
            critique.append(f"{name}={value}")
    critique.extend(policy.get("reasons", []))

    print(json.dumps({
        "action": action,
        "probabilities": {
            name: a.get("probabilities") for name, a in answers.items()
        },
        "reasons": policy.get("reasons", []),
        "verdict": answers.get("acceptable", {}).get("probabilities"),
    }, ensure_ascii=False))

    return {"accept": 0, "revise": 1, "abstain": 2}.get(action, 2)


if __name__ == "__main__":
    raise SystemExit(main())
