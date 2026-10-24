"""Labeled-set evaluation: find out what the model is good at, per question type.

Input: a questions spec + examples of {state, labels}. Labels:
    choice/multi/tool  — the correct option key        -> top-1 accuracy + confusion
    noul               — true / false                  -> accuracy, false positives/negatives
    score              — a number (the ideal anchor)   -> mean absolute error

Plus probability calibration: bucketed predicted confidence vs observed accuracy.
Rollout ladder: shadow (log only) -> suggestion -> low-risk automation.
"""

from __future__ import annotations

import math
from typing import Any

CALIBRATION_EDGES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _p_true(probs: Any) -> float | None:
    """P(true) from an answer's probabilities dict (case-tolerant)."""
    if not isinstance(probs, dict):
        return None
    for key in ("true", "True", "TRUE"):
        value = probs.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def evaluate_question(
    qname: str,
    qtype: str,
    pairs: list[tuple[dict[str, Any], Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """pairs of (answer, label) for one question."""
    n = len(pairs)
    if n == 0:
        return {"type": qtype, "n": 0}

    report: dict[str, Any] = {"type": qtype, "n": n}

    if qtype in ("choice", "multi", "tool"):
        correct = 0
        confusion: dict[str, dict[str, int]] = {}
        for answer, label in pairs:
            predicted = answer.get("choice")
            key = str(label)
            slot = confusion.setdefault(key, {})
            slot[str(predicted)] = slot.get(str(predicted), 0) + 1
            if predicted == label:
                correct += 1
        report["accuracy"] = round(correct / n, 3)
        report["confusion"] = confusion

    elif qtype == "noul":
        tp = tn = fp = fn = 0
        for answer, label in pairs:
            p_true = _p_true(answer.get("probabilities"))
            predicted_true = (p_true if p_true is not None else 0.0) >= threshold
            label_true = bool(label)
            if predicted_true and label_true:
                tp += 1
            elif predicted_true and not label_true:
                fp += 1
            elif not predicted_true and not label_true:
                tn += 1
            else:
                fn += 1
        report.update(
            accuracy=round((tp + tn) / n, 3),
            true_positives=tp, true_negatives=tn,
            false_positives=fp, false_negatives=fn,
            threshold=threshold,
        )

    elif qtype == "score":
        errors = []
        for answer, label in pairs:
            value = _parse_float(answer.get("score", answer.get("choice")))
            target = _parse_float(label)
            if value is not None and target is not None:
                errors.append(abs(value - target))
        if errors:
            report["mae"] = round(sum(errors) / len(errors), 3)
            report["max_error"] = round(max(errors), 3)
            report["scored"] = len(errors)
        else:
            report["mae"] = None

    # calibration buckets from confidence
    buckets: dict[int, dict[str, float]] = {}
    for answer, label in pairs:
        conf = answer.get("confidence")
        if not isinstance(conf, (int, float)):
            continue
        if qtype == "choice":
            hit = 1.0 if answer.get("choice") == label else 0.0
        elif qtype == "noul":
            p_true = _p_true(answer.get("probabilities"))
            hit = 1.0 if ((p_true if p_true is not None else 0.0) >= threshold) == bool(label) else 0.0
        else:
            continue
        idx = next((i for i, edge in enumerate(CALIBRATION_EDGES) if conf < edge), len(CALIBRATION_EDGES) - 1)
        bucket = buckets.setdefault(idx, {"sum_conf": 0.0, "sum_hit": 0.0, "n": 0})
        bucket["sum_conf"] += conf
        bucket["sum_hit"] += hit
        bucket["n"] += 1
    if buckets:
        report["calibration"] = [
            {
                "confidence_range": (
                    f"{CALIBRATION_EDGES[i-1]:.1f}-{CALIBRATION_EDGES[i]:.1f}"
                    if i > 0 else f"<{CALIBRATION_EDGES[0]:.1f}"
                ),
                "n": b["n"],
                "mean_confidence": round(b["sum_conf"] / b["n"], 3),
                "observed_accuracy": round(b["sum_hit"] / b["n"], 3),
                "gap": round(abs(b["sum_conf"] / b["n"] - b["sum_hit"] / b["n"]), 3),
            }
            for i, b in sorted(buckets.items())
        ]
    return report


def evaluate(
    questions: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """results[i]/labels[i] are the answer dict / label dict for example i."""
    per_question: dict[str, list[tuple[dict[str, Any], Any]]] = {
        qname: [] for qname in questions
    }
    for result, label in zip(results, labels):
        answers = result.get("answers", {})
        for qname, spec in questions.items():
            answer = answers.get(qname)
            label_value = label.get(qname)
            if answer is None or label_value is None:
                continue
            per_question[qname].append((answer, label_value))

    report: dict[str, Any] = {"questions": {}, "n_examples": len(results)}
    accuracies = []
    for qname, spec in questions.items():
        qtype = spec.get("type", "choice")
        q_report = evaluate_question(qname, qtype, per_question[qname], threshold)
        report["questions"][qname] = q_report
        if isinstance(q_report.get("accuracy"), (int, float)) and not math.isnan(q_report["accuracy"]):
            accuracies.append(q_report["accuracy"])
        if q_report.get("mae") is not None:
            accuracies.append(max(0.0, 1.0 - q_report["mae"]))  # count MAE in overall as inverted skill
    if accuracies:
        report["overall"] = round(sum(accuracies) / len(accuracies), 3)
    return report
