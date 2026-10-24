"""Decision policies: the model returns signals; code owns the consequences.

A policy turns a decision result into one action plus the reasons for it.
Principles:
    - model output is a signal, never a permission token
    - low confidence, deferrals and failed checks must fall back to a safe path
    - every automated action can name why it happened (audit trail)

Precedence: deferral > failed noul/score checks > low confidence > route
mapping > pass action > fallback. Checks always override routes — fail closed.
`pass_action` names the automation for the all-checks-passed case (gates);
`fallback` stays the safe path for unmatched outcomes.
"""

from __future__ import annotations

from typing import Any


def _score_value(answer: dict[str, Any]) -> float | None:
    """Numeric value of a score answer.

    The engine emits a continuous `score` field (not necessarily anchored to
    the criteria keys); older shapes fall back to the chosen anchor key.
    """
    for key in ("score", "value", "choice"):
        value = answer.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def apply_policy(answers: dict[str, dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Interpret a decision result against a policy spec.

    Policy spec keys (all optional):
        routes           map "<question>.<choice>" (or bare "<choice>") -> action
        min_confidence   below this per-answer confidence -> defer_action
        defer_action     action when the model deferred or confidence is low
        noul_checks      {q: {"expect": true/false, "min": p}}
        score_checks     {q: {"max"|"min": value}}
        check_action     action when a noul/score check fails (default: fallback)
        pass_action      action when every check passed and no route matched
        fallback         action when nothing else matched (required)
    """
    fallback = policy.get("fallback", "human_review")
    defer_action = policy.get("defer_action", fallback)
    check_action = policy.get("check_action", fallback)
    routes: dict[str, str] = policy.get("routes", {})
    min_confidence = policy.get("min_confidence")

    reasons: list[str] = []
    deferred = False
    violated = False
    route_action: str | None = None

    for qname, answer in answers.items():
        conf = answer.get("confidence")

        if answer.get("deferred"):
            deferred = True
            reasons.append(f"{qname}: model deferred (below certified threshold)")
        elif min_confidence is not None and isinstance(conf, (int, float)) and conf < min_confidence:
            deferred = True
            reasons.append(f"{qname}: confidence {conf:.2f} < {min_confidence}")

        atype = answer.get("type")
        if atype == "choice":
            choice = answer.get("choice")
            if choice is not None and route_action is None:
                mapped = routes.get(f"{qname}.{choice}", routes.get(choice))
                if mapped is not None:
                    route_action = mapped
        elif atype == "noul":
            checks = policy.get("noul_checks", {}).get(qname)
            if checks is not None:
                expect = bool(checks.get("expect", True))
                min_p = float(checks.get("min", 0.5))
                probs = answer.get("probabilities", {})
                p = probs.get("true" if expect else "false")
                if p is None:
                    p = probs.get(str(expect))
                if p is not None and p < min_p:
                    violated = True
                    reasons.append(
                        f"{qname}: expected {'true' if expect else 'false'} "
                        f"at >= {min_p}, got {p:.2f}"
                    )
        elif atype == "score":
            checks = policy.get("score_checks", {}).get(qname)
            value = _score_value(answer)
            if checks is not None and value is not None:
                if "max" in checks and value > float(checks["max"]):
                    violated = True
                    reasons.append(f"{qname}: score {value} > max {checks['max']}")
                if "min" in checks and value < float(checks["min"]):
                    violated = True
                    reasons.append(f"{qname}: score {value} < min {checks['min']}")

    if deferred:
        action = defer_action
        reasons.append(f"deferred/low-confidence -> {defer_action}")
    elif violated:
        action = check_action
        reasons.append(f"check failed -> {check_action}")
    elif route_action is not None:
        action = route_action
        reasons.append(f"route match -> {action}")
    elif "pass_action" in policy:
        action = policy["pass_action"]
        reasons.append(f"all checks passed -> {action}")
    else:
        action = fallback
        reasons.append(f"no rule matched -> fallback {fallback}")

    confidences = [
        a.get("confidence") for a in answers.values()
        if isinstance(a.get("confidence"), (int, float))
    ]
    return {
        "action": action,
        "flagged": action in (defer_action, check_action, fallback),
        "reasons": reasons,
        "min_confidence": round(min(confidences), 3) if confidences else None,
    }
