from basal_station.recipes import RECIPES, get_recipe, list_recipes, recipe_payload

VALID_TYPES = {"choice", "noul", "score", "multi", "act", "tool"}
ROLES = {"triage", "router", "gate", "judge", "ground"}


def test_recipe_keys_unique():
    keys = [r.key for r in RECIPES]
    assert len(keys) == len(set(keys))


def test_every_recipe_is_wellformed():
    for r in list_recipes():
        assert r.role in ROLES, r.key
        assert r.questions and r.sample_state.strip(), r.key
        assert r.description.strip(), r.key
        for qname, spec in r.questions.items():
            assert spec["type"] in VALID_TYPES, (r.key, qname)
            if spec["type"] in ("choice", "multi", "tool", "score"):
                assert len(spec.get("criteria", {})) >= 2, (r.key, qname)
        if r.policy is not None:
            assert "fallback" in r.policy, r.key


def test_notable_recipes_present():
    keys = {r.key for r in list_recipes()}
    assert {"triage", "router", "safety_gate", "answer_quality",
            "evidence_sufficiency"} <= keys


def test_get_recipe_and_payload():
    r = get_recipe("triage")
    assert r is not None
    assert get_recipe("nope") is None
    payload = recipe_payload(r)
    assert payload["key"] == "triage"
    assert "route" in payload["questions"]
    assert payload["policy"]["fallback"]
