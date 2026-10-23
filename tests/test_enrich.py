from basal_station.knowledge.enrich import (
    auto_query,
    format_hits_block,
    ground_state,
    hits_summary,
)


def test_auto_query_uses_names_and_state():
    q = auto_query(
        "KYC escalation for a business account with PEP match",
        {"routing": {"type": "choice", "instructions": "route the escalation",
                     "criteria": {"edd": "enhanced due diligence"}}},
    )
    assert "routing" in q
    assert "route the escalation" in q
    assert "edd" in q
    assert "PEP" in q


def test_format_hits_block_numbered_with_sources():
    payload = [
        {"source": "AML playbook.md", "text": "PEP matches above 70% require EDD."},
        {"title": "policy", "text": "Never auto-clear PEP hits."},
        {"text": ""},
    ]
    block, hits = format_hits_block("pep policy", payload)
    assert "[1] (AML playbook.md) PEP matches above 70%" in block
    assert "[2] (policy) Never auto-clear" in block
    assert len(hits) == 2  # empty-text hit dropped


def test_format_hits_block_handles_plain_text_and_none():
    assert format_hits_block("q", None)[0] == ""
    block, hits = format_hits_block("q", "just a string")
    assert "[1] just a string" in block
    assert hits[0]["text"] == "just a string"
    block2, _ = format_hits_block("q", {"results": [{"text": "res"}]})
    assert "[1] res" in block2


def test_ground_state_appends_block():
    grounded, hits = ground_state("Raw state.", [{"source": "kb", "text": "fact"}], "q")
    assert grounded.startswith("Raw state.")
    assert "Reference knowledge" in grounded
    assert "[1] (kb) fact" in grounded
    assert len(hits) == 1


def test_hits_summary_caps():
    payload = [{"text": f"t{i}"} for i in range(20)]
    assert len(hits_summary(payload, 5)) == 5
