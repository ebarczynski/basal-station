"""Regression tests for the grill's small-batch fixes (queue item 7)."""

from basal_station.api.downloads import _quant_matches
from basal_station.eval import evaluate_question
from basal_station.knowledge.enrich import MAX_BLOCK_CHARS, format_hits_block


def test_enrich_budget_is_hard_cap():
    """The negative-slice bug let a 6000-char budget grow to 6923 (measured)."""
    payload = [{"text": "x" * 900, "source": f"doc{i}"} for i in range(12)]
    block, _kept = format_hits_block("q", payload)
    assert len(block) <= MAX_BLOCK_CHARS


def test_quant_token_match_prefers_exact_file(monkeypatch):
    assert _quant_matches("model-Q4_K_M.gguf", "Q4_K_M")
    assert not _quant_matches("model-XQ4_K_M.gguf", "Q4_K_M")  # no token boundary

    import huggingface_hub

    files = ["model-XQ4_K_M.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"]
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo: files)

    from basal_station.api.downloads import _pick_gguf_files

    picked = _pick_gguf_files("repo", "Q4_K_M")
    assert picked == ["model-Q4_K_M.gguf"]


def test_eval_noul_case_tolerant_p_true():
    """The old fallback str(bool(label)) produced 'True' which never matched
    JSON keys, silently scoring examples as predicted-false."""
    pairs = [({"probabilities": {"True": 0.9}}, True)]  # JSON-bool-keyed shape
    report = evaluate_question("q", "noul", pairs, threshold=0.5)
    assert report["true_positives"] == 1
    assert report["false_negatives"] == 0

    pairs_lower = [({"probabilities": {"true": 0.9}}, True)]
    report2 = evaluate_question("q", "noul", pairs_lower, threshold=0.5)
    assert report2["true_positives"] == 1
