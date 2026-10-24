from basal_station.policy import apply_policy

ANSWERS = {
    "route": {
        "type": "choice", "choice": "billing",
        "probabilities": {"billing": 0.7, "technical": 0.2},
        "confidence": 0.7,
    },
    "impact": {
        "type": "score", "choice": "0.3",
        "probabilities": {"0.0": 0.2, "0.3": 0.6, "1.0": 0.2},
        "confidence": 0.6,
    },
    "needs_human": {
        "type": "noul", "probabilities": {"true": 0.1, "false": 0.9},
        "confidence": 0.9,
    },
}

POLICY = {
    "routes": {"route.billing": "queue_billing", "route.technical": "queue_tech"},
    "score_checks": {"impact": {"max": 0.6}},
    "noul_checks": {"needs_human": {"expect": False, "min": 0.7}},
    "check_action": "human_review",
    "min_confidence": 0.6,
    "defer_action": "human_review",
    "fallback": "human_review",
}


def test_policy_routes_confident_choice():
    out = apply_policy(ANSWERS, POLICY)
    assert out["action"] == "queue_billing"
    assert not out["flagged"]
    assert any("route match" in r for r in out["reasons"])
    assert out["min_confidence"] == 0.6


def test_policy_score_violation_overrides():
    answers = {**ANSWERS, "impact": {**ANSWERS["impact"], "choice": "1.0"}}
    out = apply_policy(answers, POLICY)
    assert out["action"] == "human_review"
    assert out["flagged"]
    assert any("score" in r for r in out["reasons"])


def test_policy_noul_violation_overrides():
    answers = {
        **ANSWERS,
        "needs_human": {**ANSWERS["needs_human"], "probabilities": {"true": 0.9, "false": 0.1}},
    }
    out = apply_policy(answers, POLICY)
    assert out["action"] == "human_review"
    assert any("expected false" in r for r in out["reasons"])


def test_policy_low_confidence_defers():
    answers = {
        **ANSWERS,
        "route": {**ANSWERS["route"], "confidence": 0.4},
    }
    out = apply_policy(answers, POLICY)
    assert out["action"] == "human_review"
    assert out["flagged"]


def test_policy_deferred_answer_defers():
    answers = {**ANSWERS, "route": {**ANSWERS["route"], "deferred": True}}
    out = apply_policy(answers, POLICY)
    assert out["action"] == "human_review"


def test_policy_fallback_when_no_route_matches():
    answers = {**ANSWERS, "route": {**ANSWERS["route"], "choice": "unknown_team"}}
    out = apply_policy(answers, POLICY)
    assert out["action"] == "human_review"


def test_policy_pass_action_when_all_checks_pass():
    answers = {"ok": {"type": "noul", "probabilities": {"true": 0.95}, "confidence": 0.95}}
    out = apply_policy(answers, {
        "noul_checks": {"ok": {"expect": True, "min": 0.8}},
        "pass_action": "auto_approve",
        "fallback": "review",
    })
    assert out["action"] == "auto_approve"
    assert not out["flagged"]
    assert any("all checks passed" in r for r in out["reasons"])


def test_policy_pass_action_not_used_when_check_fails():
    answers = {"ok": {"type": "noul", "probabilities": {"true": 0.5}, "confidence": 0.5}}
    out = apply_policy(answers, {
        "noul_checks": {"ok": {"expect": True, "min": 0.8}},
        "check_action": "review",
        "pass_action": "auto_approve",
        "fallback": "review",
    })
    assert out["action"] == "review"
    assert out["flagged"]


def test_policy_default_fallback_without_spec():
    out = apply_policy({"q": {"type": "choice", "choice": "x", "confidence": 0.9}}, {})
    assert out["action"] == "human_review"
